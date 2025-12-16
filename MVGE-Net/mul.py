def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}
        #return {field: torch.exp((edges.data[field] / scale_constant))}
    return func
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias = True):
        super().__init__()
        #in_dim:100
        #out_dim:10
        self.out_dim = out_dim #10
        self.num_heads = num_heads#10
        num_steps = 6
        max_edge_types = 5
        self.feature_Q = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, self_loop = False, dropout = 0.1)
        self.feature_K = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, self_loop = False, dropout = 0.1)
        self.feature_V = RelGraphConv(in_feat=in_dim, out_feat=in_dim, num_rels = 9, regularizer='basis', num_bases = 9, activation = f.relu, self_loop = False, dropout = 0.1)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    # attn_out = self.attention(graph, h, e)
    # self.attention = MultiHeadAttentionLayer(input_dim, output_dim // num_heads, num_heads, use_bias)
    # attn_out = self.attention(graph, h, e)
    def forward(self, g, h, e):
        # g:graph
        # h:features
        # e:edge_type

        feature_Q = self.feature_Q(g, h, e)
        #h形状为 num_nodes*100
        #feature_Q = num_nodes*100
        # self.feature_Q = RelGraphConv(in_feat=100, out_feat=100, num_rels=9, regularizer='basis', num_bases=9,
        #                               activation=f.relu, self_loop=False, dropout=0.1)
        # self.feature_K = RelGraphConv(in_feat=100, out_feat=100, num_rels=9, regularizer='basis', num_bases=9,
        #                               activation=f.relu, self_loop=False, dropout=0.1)
        # self.feature_V = RelGraphConv(in_feat=100, out_feat=100, num_rels=9, regularizer='basis', num_bases=9,
        #                               activation=f.relu, self_loop=False, dropout=0.1)
        #print(feature_Q.shape)
        feature_K = self.feature_K(g, h, e)
        feature_V = self.feature_V(g, h, e)
        #print(feature_Q.shape)
        Q_h = feature_Q#self.Q(feature_Q)
        K_h = feature_K#self.K(feature_K)
        V_h = feature_V#self.V(feature_V)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        #Q_h:num_nodes *100
        #g.ndata['Q_h']:num_nodes*100 -> -1 * [num_heads(10) * out_dim(10)](100) 
        #-1:num_nodes
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)#
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)#
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)#

        self.propagate_attention(g)

        #head_out = g.ndata['wV'] / g.ndata['z']
        head_out= g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        return head_out