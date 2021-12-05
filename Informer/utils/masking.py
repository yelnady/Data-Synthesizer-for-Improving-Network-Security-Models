import torch

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # B is batch size, H is n_heads, L is how many timesteps (Length)
        # scores.shape[-1] is 400
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
    
class MyProbMask():
    def __init__(self, B, H, L, index, scores,attn_mask,device="cpu"):
        
        # attn_mask passed is already a ready matrix that has the True and False for self-attention mask + padding mask
        # B is batch size, H is n_heads, L is how many timesteps (Length)
        # scores.shape[-1] is 400
        
        if attn_mask is not None:
            _mask_ex = attn_mask[:,None,:,:].expand(B, H, L, scores.shape[-1])
        else: #Generating
            _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
            _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])

        
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask