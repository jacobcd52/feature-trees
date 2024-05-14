#%%
from huggingface_hub import snapshot_download
snapshot_download(repo_id="pchlenski/gpt2-transcoders", allow_patterns=["*.pt"],
    local_dir="/workspace/transcoder_circuits/gpt-2-small-transcoders", local_dir_use_symlinks=False
)

#%%
import collections
from functools import partial
from jaxtyping import Float
from torch import Tensor
from einops import *
import transformer_lens as tl
import plotly.express as px
import gc
import inspect
model = tl.HookedTransformer.from_pretrained("gpt2").cuda()

#%%
from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
from sae_training.sparse_autoencoder import SparseAutoencoder
transcoder_template = "/workspace/transcoder_circuits/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
transcoders = []
for i in range(model.cfg.n_layers):
    transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval().cuda())

#%%
from transcoder_circuits import load_kissane
attn_saes_dic = load_kissane.get_saes()
attn_saes = [attn_saes_dic[f'blocks.{layer}.attn.hook_z'] for layer in range(model.cfg.n_layers)]
for sae in attn_saes:
    sae.W_enc = torch.nn.Parameter(sae.W_enc[:, :24575], requires_grad=False)
    sae.b_enc = torch.nn.Parameter(sae.b_enc[:24575], requires_grad=False)
    sae.W_dec = torch.nn.Parameter(sae.W_dec[:24575, :], requires_grad=False)
    sae.b_dec = torch.nn.Parameter(sae.b_dec, requires_grad=False)

del attn_saes_dic
gc.collect()
torch.cuda.empty_cache()
# %%
# useful debug function
def help(var_list):
    for var in var_list:
        # Retrieve the caller's frame and extract variable names and values
        frame = inspect.currentframe().f_back
        name = None
        for var_name, var_val in frame.f_locals.items():
            if var_val is var:
                name = var_name
                break
        print(name, " : ", [i for i in var.shape])


# ----------------------- END SETUP -----------------------------
















#%%
def metric(model, tokens): 
    logits = model(tokens, return_type="logits")[:, -2, :]
    correct_logits = logits[torch.arange(logits.size(0)), tokens[:, -1]]
    
    return correct_logits.mean()



#%%
@dataclass
class Config:
    threshold : float = 0.01
    feature_act_gate_eps : float = 0.1


#%%
class CircuitFinder:
    # TODO
    # - error attributions (these might give a measure of SAE/circuit quality!)
    # - chained attributions
    # - add embeddings as upstream node
    # - mlp_n sees attn_n
    #
    #    
    def __init__(self, cfg, tokens, model, attn_saes, transcoders):
        self.cfg = cfg
        self.tokens = tokens
        self.model = model
        self.attn_saes = attn_saes
        self.transcoders = transcoders

        # Store some params for convenience
        self.d_trans = transcoders[0].W_enc.size(1)
        self.d_sae = attn_saes[0].W_enc.size(1)
        self.batch, self.n_seq = tokens.size()
        self.d_model = model.cfg.d_model
        self.n_layers = model.cfg.n_layers
        self.layers = range(model.cfg.n_layers)

        # Cache feature acts, pattern, layernorm scales, SAE errors, active feature IDs
        self.get_initial_cache()

        # Kissane et al's attention SAEs are trained at hook_z, concatenated across head_idx dimension.
        # Here we process the decoder matrices so their columns can be viewed as residual vectors.
        attn_all_W_dec_z_cat = torch.stack([attn_saes[i].W_dec for i in self.layers]) # [layer d_sae d_model]
        attn_all_W_dec_z = rearrange(attn_all_W_dec_z_cat, "layer d_sae (n_heads d_head) -> layer d_sae n_heads d_head", n_heads=model.cfg.n_heads)
        self.attn_all_W_dec_resid = einsum(attn_all_W_dec_z, model.W_O, "layer d_sae n_heads d_head , layer n_heads d_head d_model -> layer d_sae d_model")
        
        # We'll store (edge, attrib) pairs here.
        # edge = (downstream_node, upstream_node)
        # node = "{module_name}.{layer}.{pos}.{feature_id}"
        self.graph = []


    def get_initial_cache(self):
        '''Run model on tokens. Grab acts at mlp_in, and run them through 
        transcoders to get all transcoder feature acts. Grab acts at hook_z,
        concatenate on head_idx dim, and run through attn_sae to get all 
        attention SAE feature acts'''
        # This code is verbose but totally trivial!
    
        # Run model and cache the acts we need
        names_filter = lambda x: (x.endswith("ln2.hook_normalized") or x.endswith("mlp_out") or x.endswith("hook_z") or x.endswith("pattern") or x.endswith("hook_scale"))
        _ , cache = model.run_with_cache(self.tokens,
                                        return_type="loss",
                                        names_filter = names_filter)
        
        # Save attention patterns (avg over batch and head) and layernorm scales (avg over batch)
        self.batchmean_headmean_pattern = torch.stack([cache["pattern", layer].mean(dim=[0,1]) for layer in self.layers]) # [layer qpos kpos]
        self.batchmean_attn_layernorm_scales = torch.stack([cache[f"blocks.{layer}.ln1.hook_scale"].mean(dim=0) for layer in self.layers]) # [layer pos]
        self.batchmean_mlp_layernorm_scales  = torch.stack([cache[f"blocks.{layer}.ln2.hook_scale"].mean(dim=0) for layer in self.layers]) # [layer pos]

        # Save feature acts
        # Initialise empty tensors to store feature acts and is_active (both batchmeaned)
        self.mlp_batchmean_feature_acts  : Float[Tensor, "seq, layer, d_trans"] = torch.empty(self.n_seq, self.n_layers, self.d_trans).cuda()
        self.mlp_batchmean_is_active     : Float[Tensor, "seq, layer, d_trans"] = torch.empty(self.n_seq, self.n_layers, self.d_trans).cuda()
        self.attn_batchmean_feature_acts : Float[Tensor, "seq, layer, d_sae"]   = torch.empty(self.n_seq, self.n_layers, self.d_sae).cuda()
        self.attn_batchmean_is_active    : Float[Tensor, "seq, layer, d_sae"]   = torch.empty(self.n_seq, self.n_layers, self.d_sae).cuda()
        self.mlp_batchmean_errors        : Float[Tensor, "seq, layer, d_model"] = torch.empty(self.n_seq, self.n_layers, self.d_model).cuda()
        self.attn_batchmean_errors       : Float[Tensor, "seq, layer, d_model"] = torch.empty(self.n_seq, self.n_layers, self.d_model).cuda()

        # Add feature acts to the empty tensors, layer by layer
        for layer in range(self.n_layers):
            mlp_in_pt   = f'blocks.{layer}.ln2.hook_normalized'
            mlp_out_pt  = f'blocks.{layer}.hook_mlp_out'
            attn_out_pt = f'blocks.{layer}.attn.hook_z'

            # Get MLP feature acts and recons errors
            mlp_recons, mlp_feature_acts  = self.transcoders[layer](cache[mlp_in_pt])[:2]
            self.mlp_batchmean_feature_acts[:, layer, :]  = mlp_feature_acts.mean(0)
            self.mlp_batchmean_is_active[:, layer, :]  = (mlp_feature_acts > self.cfg.feature_act_gate_eps).float().mean(0)
            self.mlp_batchmean_errors[:, layer, :] = (cache[mlp_out_pt] - mlp_recons).mean(0)

            # Get attention feature acts and recons errors (remember to be careful with z concatenation!)
            z_concat = rearrange(cache[attn_out_pt], "batch seq n_heads d_head -> batch seq (n_heads d_head)")
            attn_recons, sae_cache = attn_saes[layer].run_with_cache(z_concat, names_filter='hook_sae_acts_post')
            attn_feature_acts = sae_cache['hook_sae_acts_post']
            self.attn_batchmean_feature_acts[:, layer, :] = attn_feature_acts.mean(0)
            self.attn_batchmean_is_active[:, layer, :] = (attn_feature_acts > self.cfg.feature_act_gate_eps).float().mean(0)
            z_error = rearrange((attn_recons - z_concat).mean(0), "seq (n_heads d_head) -> seq n_heads d_head", n_heads=self.model.cfg.n_heads)
            resid_error = einsum(z_error, model.W_O[layer], "seq n_heads d_heads, n_heads d_head d_model -> seq d_model")
            self.attn_batchmean_errors[:, layer, :] = resid_error
        
        # Get ids of active features
        self.mlp_active_feature_ids  = torch.where(self.mlp_batchmean_is_active.sum(0) > 0)
        self.attn_active_feature_ids = torch.where(self.attn_batchmean_is_active.sum(0) > 0)


    def get_metric_attribs(self):
        '''Step 0 of circuit discovery: get attributions from each node to the metric.

        TODO currently, if metric depends on multiple token positions, this will
        sum the gradient over those positions. Do we want to be more general, i.e.
        store separate gradients at each position? Or maybe we don't care... '''
        
        model.blocks[model.cfg.n_layers-1].mlp.b_out.grad = None
        m = metric(self.model, self.tokens)
        m.backward() # TODO don't actually need backward through whole model. If m is linear, we can disable autograd!

        # Sneaky way to get d(metric)/d(resid_post_final)
        grad = model.blocks[model.cfg.n_layers-1].mlp.b_out.grad.unsqueeze(0)
        
        # Compute attrib = upstream_feature_act * d(metric)/(d_resid_post_final) dot upstream_decoder_col
        mlp_active_W_dec, mlp_up_active_layers, mlp_up_active_feature_ids = self.get_active_mlp_W_dec(self.n_layers)
        mlp_active_batchmean_feature_acts = self.mlp_batchmean_feature_acts[-2, mlp_up_active_layers, mlp_up_active_feature_ids]
        mlp_attribs   = einsum(mlp_active_batchmean_feature_acts,
                              mlp_active_W_dec,
                              grad, 
                              "up_active_id, up_active_id d_model, imp_id d_model -> imp_id up_active_id") # imp_id is just a singleton index here, present for consistency with later stuff
        
        attn_active_W_dec, attn_up_active_layers, attn_up_active_feature_ids = self.get_active_attn_W_dec(self.n_layers)
        attn_active_batchmean_feature_acts = self.attn_batchmean_feature_acts[-2, attn_up_active_layers, attn_up_active_feature_ids]
        attn_attribs  = einsum(attn_active_batchmean_feature_acts,
                              attn_active_W_dec,
                              grad, 
                              "up_active_id, up_active_id d_model, imp_id d_model -> imp_id up_active_id")

        # Add (edge, attrib) pairs to graph, where edge = ("metric", upstream_node)
        self.add_to_graph( mlp_attribs, [0], [-2], "metric", self.n_layers, "mlp", mlp_up_active_layers, mlp_up_active_feature_ids )  #TODO generalize from pos=-1
        self.add_to_graph(attn_attribs, [0], [-2], "metric", self.n_layers, "attn", attn_up_active_layers, attn_up_active_feature_ids)

        
    def get_active_mlp_W_dec(self, down_layer):
        '''so we don't have to dot with *every* upstream feature'''
        mlp_up_active_layers = self.mlp_active_feature_ids[0][self.mlp_active_feature_ids[0]<down_layer]
        mlp_up_active_feature_ids = self.mlp_active_feature_ids[1][self.mlp_active_feature_ids[0]<down_layer]
        mlp_active_W_dec = torch.stack([self.transcoders[i].W_dec for i in range(down_layer)])[mlp_up_active_layers, mlp_up_active_feature_ids, :]
        return mlp_active_W_dec, mlp_up_active_layers, mlp_up_active_feature_ids
    
    def get_active_attn_W_dec(self, down_layer):
        attn_up_active_layers = self.attn_active_feature_ids[0][self.attn_active_feature_ids[0]<down_layer]
        attn_up_active_feature_ids = self.attn_active_feature_ids[1][self.attn_active_feature_ids[0]<down_layer]
        attn_active_W_dec = self.attn_all_W_dec_resid[attn_up_active_layers, attn_up_active_feature_ids, :]
        return attn_active_W_dec, attn_up_active_layers, attn_up_active_feature_ids
    

    def mlp_step(self, down_layer):
        '''For each important node at this MLP, compute attrib wrt all previous nodes'''

        # Get the important features coming out of the MLP, and the positions at which they're important
        important_down_feature_ids, important_down_pos = self.get_important_feature_ids_and_pos("mlp", down_layer)
        # For each important downstream (feature_id, pos) pair, get batchmeaned is_active, and the corresponding encoder row
        important_active = self.mlp_batchmean_is_active[important_down_pos, down_layer, important_down_feature_ids] # [imp_id]
        important_enc_cols = self.transcoders[down_layer].W_enc[:, important_down_feature_ids] # [d_model, imp_id]
        important_layernorm_scales = self.batchmean_mlp_layernorm_scales[down_layer, important_down_pos] # [imp_id, 1]

        # The grad of these important feature acts is just the corresponding row of W_enc (scaled by layernorm)
        grad = einsum(important_active, important_enc_cols, "imp_id, d_model imp_id -> imp_id d_model")
        grad /= important_layernorm_scales

        # ----- upstream MLP ATTRIBS ------
        # Get upstream feature acts, since we'll need this for attrib computation
        mlp_active_W_dec, mlp_up_active_layers, mlp_up_active_feature_ids = self.get_active_mlp_W_dec(down_layer)
        important_mlp_batchmean_feature_acts = self.mlp_batchmean_feature_acts[:, mlp_up_active_layers, mlp_up_active_feature_ids] # [imp_id, up_active_id]
        important_mlp_batchmean_feature_acts = important_mlp_batchmean_feature_acts[important_down_pos]
        # Compute attribs of important nodes wrt all upstream MLP nodes
        mlp_attribs   = einsum(important_mlp_batchmean_feature_acts,
                              mlp_active_W_dec,
                              grad, 
                              "imp_id up_active_id, up_active_id d_model, imp_id d_model -> imp_id up_active_id")
        # attrib can be at most the value of the original downstream feature act
        #mlp_attribs = torch.min(mlp_attribs, important_mlp_batchmean_feature_acts)
        self.add_to_graph(mlp_attribs, important_down_feature_ids, important_down_pos, down_module_name="mlp", down_layer=down_layer, 
                          up_module_name="mlp", up_active_layers=mlp_up_active_layers, up_active_feature_ids=mlp_up_active_feature_ids)     

        del mlp_attribs
        gc.collect()
        torch.cuda.empty_cache()

        # ----- upstream ATTENTION ATTRIBS (do exactly the same for attention) -----
        attn_active_W_dec, attn_up_active_layers, attn_up_active_feature_ids = self.get_active_attn_W_dec(down_layer)
        important_attn_batchmean_feature_acts = self.attn_batchmean_feature_acts[:, attn_up_active_layers, attn_up_active_feature_ids] # [imp_id, up_active_id]
        important_attn_batchmean_feature_acts = important_attn_batchmean_feature_acts[important_down_pos]
        attn_attribs   = einsum(important_attn_batchmean_feature_acts,
                                attn_active_W_dec,
                                grad, 
                               "imp_id up_active_id, up_active_id d_model, imp_id d_model -> imp_id up_active_id")
        #attn_attribs = torch.min(attn_attribs, important_attn_batchmean_feature_acts)       
        self.add_to_graph(attn_attribs, important_down_feature_ids, important_down_pos, down_module_name="mlp", down_layer=down_layer, 
                          up_module_name="attn", up_active_layers=attn_up_active_layers, up_active_feature_ids=attn_up_active_feature_ids)     
        
        del attn_attribs
        gc.collect()
        torch.cuda.empty_cache()         


    def ov_step(self, down_layer):
        '''For each important node at this attention layer, compute attrib wrt all previous nodes *via the OV circuit* '''

        # Get the important features coming out of the attention layer, and the positions at which they're important
        important_down_feature_ids, important_down_pos = self.get_important_feature_ids_and_pos("attn", down_layer)
        
        # For each important downstream (feature_id, pos) pair, get batchmeaned is_active, encoder row, and pattern
        important_active = self.attn_batchmean_is_active[important_down_pos, down_layer, important_down_feature_ids] # [imp_id]
        important_enc_rows = self.attn_saes[down_layer].W_enc[:, important_down_feature_ids] # [d_model, imp_id]
        important_patterns = self.batchmean_headmean_pattern[down_layer, important_down_pos] # [imp_id, kpos]
        important_layernorm_scales = self.batchmean_attn_layernorm_scales[down_layer, important_down_pos].unsqueeze(2) # [imp_id, 1, 1]
        
        # ..
        W_V_concat = rearrange(model.W_V[down_layer], "head_id d_model d_head -> d_model (head_id d_head)")
        grad_pre = einsum(important_active, important_enc_rows, W_V_concat, important_patterns,
                          "imp_id, concat imp_id, d_model concat, imp_id kpos -> imp_id kpos d_model")
        grad_pre /= important_layernorm_scales

        
        # ----- MLP ATTRIBS ------
        # Get all upstream feature acts. Note that unlike mlp_step, we need upstream feature acts at all positions
        mlp_active_W_dec, mlp_up_active_layers, mlp_up_active_feature_ids = self.get_active_mlp_W_dec(down_layer)
        important_mlp_batchmean_feature_acts = self.mlp_batchmean_feature_acts[:, mlp_up_active_layers, mlp_up_active_feature_ids] # [imp_id, up_active_id]
        important_mlp_batchmean_feature_acts = important_mlp_batchmean_feature_acts[important_down_pos]
               
        # Compute attribs of important nodes wrt all upstream MLP nodes
        mlp_attribs   = einsum(important_mlp_batchmean_feature_acts,
                              mlp_active_W_dec,
                              grad_pre, 
                              "up_seq up_active_id, up_active_id d_model, imp_id kpos d_model -> imp_id kpos up_active_id")
        # attrib can be at most the value of the original downstream feature act
        #mlp_attribs = torch.min(mlp_attribs, self.mlp_batchmean_feature_acts[important_down_pos, down_layer].unsqueeze(1).unsqueeze(2))

        self.add_to_graph(mlp_attribs, important_down_feature_ids, important_down_pos, down_module_name="attn", down_layer=down_layer, 
                          up_module_name="mlp", up_active_layers=mlp_up_active_layers, up_active_feature_ids=mlp_up_active_feature_ids)     

        del mlp_attribs
        gc.collect()
        torch.cuda.empty_cache()

        # ----- ATTN ATTRIBS ------
        # Get all upstream feature acts. Note that unlike mlp_step, we need upstream feature acts at all positions
        attn_active_W_dec, attn_up_active_layers, attn_up_active_feature_ids = self.get_active_attn_W_dec(down_layer)
        important_attn_batchmean_feature_acts = self.attn_batchmean_feature_acts[:, attn_up_active_layers, attn_up_active_feature_ids] # [imp_id, up_active_id]
        important_attn_batchmean_feature_acts = important_attn_batchmean_feature_acts[important_down_pos]
        # Compute attribs of important nodes wrt all upstream attn nodes
        attn_attribs   = einsum(important_attn_batchmean_feature_acts,
                              attn_active_W_dec,
                              grad_pre, 
                              "up_seq up_active_id, up_active_id d_model, imp_id kpos d_model -> imp_id kpos up_active_id")
        # attrib can be at most the value of the original downstream feature act
        #attn_attribs = torch.min(attn_attribs, self.attn_batchmean_feature_acts[important_down_pos, down_layer].unsqueeze(1).unsqueeze(2))

        self.add_to_graph(attn_attribs, important_down_feature_ids, important_down_pos, down_module_name="attn", down_layer=down_layer, 
                          up_module_name="attn", up_active_layers=attn_up_active_layers, up_active_feature_ids=attn_up_active_feature_ids)     
        
        del attn_attribs
        gc.collect()
        torch.cuda.empty_cache()


    def get_important_feature_ids_and_pos(self, module, layer):
        important_feature_ids = []
        important_pos = []
        for (down_node, up_node), attrib in self.graph:
            module_, layer_, pos, feature_id = up_node.split('.')
            if module_ == module and layer_ == str(layer):
                important_feature_ids += [int(feature_id)]
                important_pos += [int(pos)]
        return important_feature_ids, important_pos
    
    
    def add_to_graph(self, attribs, important_down_feature_ids, important_down_pos, down_module_name, down_layer, 
                     up_module_name, up_active_layers, up_active_feature_ids):
        # Convert lists to PyTorch tensors
        important_down_feature_ids = torch.tensor(important_down_feature_ids, dtype=torch.long, device=attribs.device)
        important_down_pos = torch.tensor(important_down_pos, dtype=torch.long, device=attribs.device)
        
        # TODO chained_attribs = # imp_id up_layer up_d_trans

        # Create a mask where attribs are greater than the threshold
        mask = attribs > self.cfg.threshold
        # Use the mask to find the relevant indices
        if down_module_name in ["mlp", "metric"]:
            imp_ids, up_active_ids = torch.where(mask)
            attrib_values = attribs[imp_ids, up_active_ids].flatten()

        if down_module_name == "attn":
            imp_ids, up_seqs, up_active_ids = torch.where(mask)
            attrib_values = attribs[imp_ids, up_seqs, up_active_ids].flatten()

        # Get corresponding down_feature_ids and seqs using tensor indexing
        down_feature_ids = important_down_feature_ids[imp_ids]
        down_seqs = important_down_pos[imp_ids]
        if down_module_name in ["mlp", "metric"]:
            up_seqs = down_seqs

        up_feature_ids = up_active_feature_ids[up_active_ids]
        up_layer_ids = up_active_layers[up_active_ids]
        
        
        # Construct edges based on the indices and mask
        edges = [
            (f"{down_module_name}.{down_layer}.{down_seqs[i]}.{down_feature_ids[i]}", 
            f"{up_module_name}.{up_layer_ids[i]}.{up_seqs[i]}.{up_feature_ids[i]}")
            for i in range(len(imp_ids))
        ]
        
        # Append to the graph
        for edge, value in zip(edges, attrib_values):
            self.graph.append((edge, value.item()))









# %%
tokens = model.to_tokens(["When John and Mary were at the park, John passed the ball to Mary"])
cfg = Config(threshold=0.2)
finder = CircuitFinder(cfg, tokens, model, attn_saes, transcoders)
#%%

finder.get_metric_attribs()
print(len(finder.graph))

#%%
for layer in reversed(range(1, model.cfg.n_layers)):
    print("layer : ", layer)
    finder.mlp_step(layer)
    print(len(finder.graph))
    finder.ov_step(layer)
    print(len(finder.graph))
# %%
finder.graph
#%%

