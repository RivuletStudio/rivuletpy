import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from config.Config_testing import Config
from package.rivuletpy.rivuletpy.utils.io import loadswc
from package.rivuletpy.rivuletpy.utils.graph_helper4 import post_processing_edge_list
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_subs_from_test_list
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_newnodes_from_swc
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_heatmap_from_casename
from package.rivuletpy.rivuletpy.utils.graph_helper4 import get_node_feature_from_heatmap
from package.rivuletpy.rivuletpy.utils.graph_helper4 import create_edge_list
from package.rivuletpy.rivuletpy.utils.graph_helper5 import get_newnodes_label


class VesselDHNet(torch.nn.Module):
    def __init__(self, num_features_input, num_classes_input):
        super(VesselDHNet, self).__init__()
        self.conv1 = GATConv(num_features_input, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes_input, heads=1, concat=True, dropout=0.6)

    def forward(self, data_input):
        x = F.dropout(data_input.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data_input.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data_input.edge_index)
        return F.log_softmax(x, dim=1)


print('---VesselDHNet Done---')

thre = 0.1
config = Config()
subs = get_subs_from_test_list(config_input=config)
sub = subs[0]
pred_swcpath = './dh_test_bk_pts_online/tracing_eval/pt_vessel_pred/fe131b5c-d32a-4a58-9214-b327021f63a8/vessel_pred_0.1.swc'
pred_swc = loadswc(pred_swcpath)
pred_swc_copy = pred_swc.copy()

newnodes = get_newnodes_from_swc(swc_array_input=pred_swc_copy)
print('newnodes.shape', newnodes.shape)
print('---New Node IDs Done---')

edge_list = create_edge_list(swc_array_input=pred_swc_copy, newnode_all_input=newnodes)
edge_list = post_processing_edge_list(edge_list)
print('---New Edge Connection Done---')

newnodes_label = get_newnodes_label(sub_input=sub)
print('newnodes_label', newnodes_label.shape)
print('---Node Label Done---')

heatmap, heatmap_bi = get_heatmap_from_casename(sub_input=sub, config_input=config)
print('---Heatmap Done---')
node_features = np.zeros((newnodes.shape[0], 27))
for node_i in range(newnodes.shape[0]):
    node_i_feature = get_node_feature_from_heatmap(newnode_id_input=node_i,
                                                   newnode_all_input=newnodes,
                                                   heatmap_input=heatmap)
    node_features[node_i, :] = node_i_feature
print('---Node Features Done---')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.from_numpy(node_features).float().to(device)
edge_index = np.transpose(edge_list)
edge_index = torch.from_numpy(edge_index).float().to(device)
edge_index = edge_index.type(torch.cuda.LongTensor)
y = torch.from_numpy(newnodes_label).float().to(device)
y = y.type(torch.cuda.LongTensor)
data2 = Data(x=x, edge_index=edge_index, y=y)
data2 = data2.to(device)
num_features = 27
num_classes = 2
model_vessel_dh = VesselDHNet(num_features_input=num_features, num_classes_input=num_classes)
model_vessel_dh = model_vessel_dh.to(device)
optimizer = torch.optim.Adam(model_vessel_dh.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(1, 10):
    model_vessel_dh.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model_vessel_dh(data2), data2.y)
    loss.backward()
    optimizer.step()
    print('epoch', epoch, 'loss', loss.item())