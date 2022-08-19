""" 
gma.py from https://github.com/zacjiang/GMA/blob/main/core/gma.py
with additional comments and changes to work with 3D correlation volumes
"""

import torch
from torch import nn, einsum
from einops import rearrange


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        """ Initializes the relative positional embedding module
            Initialize relative height and width embedding and relative indices into the embeddings

        Args:
            max_pos_size (int): "radius" of the embedding in the p_infinity norm
            dim_head (int): dimension of the query and key vectors
        """
        super().__init__()

        # look up tables for max_pos_size indices around arbitrary pixel in height/width-direction
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        # maps two pixel indices in one dimension to their displacement
        # 2d tensor of shape (max_pos_size, max_pos_size) where t[i,j] = j-i
        # intuition: t[pixel_index_img1, corresponding_index_img1] = corresponding_index_img1 - pixel_index_img1
        # example for (max_pos_size=4):
        # tensor(  [[ 0,  1,  2,  3],
        #           [-1,  0,  1,  2],
        #           [-2, -1,  0,  1],
        #           [-3, -2, -1,  0]])
        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        
        # move value range from [-max_pos_size-1..max_pos_size-1] -> [0..2*max_pos_size-2]
        # this is done because the indices of the embeddings start at zero
        rel_ind = deltas + max_pos_size - 1
        
        # TODO: why add this as buffer? (tensor becomes part of state dict of model)
        #       rel_ind is overwritten when loading state_dict
        #       maybe as some sort of in bounds check because of limited trained embedding size
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        """ Given two pixels (x,y) and (u,v) in image1:
            Displacement (dx,dy) := (rel_ind(u), rel_ind(v)) = (u-x, v-y)
            Compute similarity between the query features at (x,y) and 
            the relative height/width embedding for (dx, dy)

        Args:
            q (torch.Tensor): query vectors for each pixel of shape (batch, heads, ht, wd, context_channels)

        Returns:
            torch.Tensor: similarity score for each pair of pixels (x,y), (u,v) in image1
        """
        batch, heads, h, w, c = q.shape

        # query height/width embedding for each pair:
        #       (index_img1.x, corresponding_index_img1.x)
        #   OR: (index_img1.y, corresponding_index_img1.y)
        # shape: (ht*ht, dim_head)
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        # shape: (wd*wd, dim_head)
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        # shape: (ht*ht, dim_head) -> (ht, ht, 1, dim_head)
        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        # shape: (wd*wd, dim_head) -> (wd, 1, wd, dim_head)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        # compute similarity between query features at pixel (x,y) in image1 and
        # displacement embedding features for (dx,dy) = (u-x, v-y) 
        # height_score[batch, head, x, y, u, v] 
        #   = query[batch, head, x, y].T @ h_emb[x,u,v(=0)]
        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        # width_score[batch, head, x, y, u, v] 
        #   = query[batch, head, x, y].T @ w_emb[y,u(=0),v]
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        # add similarity scores for height and width to get overall similarity
        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        """ Calculates the key and query features and returns their similarity
            The similarity measure is defined by the dot product of features
            The features of each pixel are learnable by parameters in this module
            Intuition:
                This module may learn query and key feature extractors such that the
                key features of pixels that are occluded are similar to query features
                of pixels that offer good replacemet motion-features

        Args:
            args (object): object with attributes holding model settings
            dim (int): dimension of the context features
            max_pos_size (int, optional): . Defaults to 100.
            heads (int, optional): number of query/key vectors per pixel. Defaults to 4.
            dim_head (int, optional): dimension of the query and key vectors. Defaults to 128.
        """
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5

        # dimension of (query/key) features of all heads concatenated
        inner_dim = heads * dim_head

        # calculates query and key features for each pixel and head
        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        
        # when position embedding is not used, pos_emb is not connected to the loss
        # this condition is added because in the above case, torch throws an exception
        if self.args.position_only or self.args.position_and_content:
            # calculates query-position similarity (position embedding used like key)
            self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        # compute query and key features from context features
        # each shape: (batch, heads*dim_head, ht, wd)
        q, k = self.to_qk(fmap).chunk(2, dim=1)

        # shape: (batch, heads*dim_head, ht, wd) -> (batch, heads, ht, wd, dim_head)
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        # compute only query-position similarity
        if self.args.position_only:
            sim = self.pos_emb(q)

        # compute query-rel_position and query-key similarity
        # add both similarity scores
        elif self.args.position_and_content:
            # query_key_similarity[batch, head, x, y, u, v]
            #   = query[batch, head, x, y].T @ key[batch, head, u, v]
            sim_content = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
            sim_pos = self.pos_emb(q)
            sim = sim_content + sim_pos

        # compute only query-key similarity
        else:
            # query_key_similarity[batch, head, x, y, u, v]
            #   = query[batch, head, x, y].T @ key[batch, head, u, v]
            sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        # shape: (batch, head, ht, wd, ht, wd) -> (batch, head, ht*wd, ht*wd)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        # softmax over all possible discrete displacements
        # all displacement scores for one pixel sum to one
        attn = sim.softmax(dim=-1)

        return attn


class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        """ aggregate motion features using global attention


        Args:
            args (object): object containing all model parameters as attributes
            dim (int): dimension of the motion features
            heads (int, optional): number of transformer heads. Defaults to 4.
            dim_head (int, optional): dimension of the value vector. Defaults to 128.
        """
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        # compute value vectors for each pixel and head
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        # if motion features dimension is not same as heads*dim_head, project using conv
        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        """ aggregate motion features using global attention

        Args:
            attn (torch.Tensor): global attention of shape (batch, heads, ht, wd, ht, wd)
            fmap (torch.Tensor): motion features of shape (batch, 126, ht, wd)

        Returns:
            torch.Tensor: learned mix of aggregated motion features and original motion features
        """
        heads, b, c, h, w = self.heads, *fmap.shape

        # values of shape (batch, heads*dim_head, ht, wd)
        v = self.to_v(fmap)
        # shape: (batch, heads*dim_head, x, y) -> (batch, heads, x*y, dim_head)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        
        # copy/mix motion features from other places in the image according to attention
        # out[batch, head, occ_pixel_idx]
        #   = sum_over(
        #       nonocc_pixel_idx,
        #       attn[batch, head, occ_pixel_idx, None, None] * v [batch, head, None, nonocc_pixel_idx]
        #       )
        # shape: (batch, heads, ht*wd, dim_head)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # shape: (batch, heads*dim_head, ht, wd)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        # do projection to match channel number of aggregated motion features with original motion features
        if self.project is not None:
            out = self.project(out)

        # mix original motion features with aggregated motion features at each pixel according to single learned ratio
        out = fmap + self.gamma * out

        return out
