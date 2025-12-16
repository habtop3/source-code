# # # dgl_bin = './dgl.bin'
# #             # with open (dgl_bin,'rb') as f:
# #             #     data = pickle.load(f)
# #             #     f.close()
# #             # a0 = graph.graph.num_nodes()
# #             # a1 = graph.graph.num_edges()
# #             # a = int(str(a0) + str(a1))
# #             # for dic in data:
# #             #     # print(a)
# #             #     b = int(list(dic.keys())[0])
# #             #     # print(b)
# #             #     if a == b:
# #             #         pass
# #             #     else:
# #             #         dic = {}
# #             #         knn_dict = load_sadj_dgl(graph)
# #             #         dic[a] = knn_dict
# #             #         with open(dgl_bin, 'rb') as file:
# #             #             data = pickle.load(file)
# #             #             file.close()
# #             #         if isinstance(data, list):
# #             # #             # 新增的字典
# #             #             new_item = dic
# #             # #             # 向列表中添加新的字典
# #             #             data.append(new_item)
# #             #         else:
# #             #             print("JSON 数据格式不正确，期望列表类型")
# #             #         with open(dgl_bin, 'wb') as file1:
# #             #             pickle.dump(data,file1)
# #             #             file1.close()
# #
# #
# #
# #             # kk
# #             # print(type(sadj))
# #             # print(sadj.shape)
# #             # print(graph.graph)
# #             # graph.graph.ndata['sadj'] = sadj
# #             # print(graph.graph)
# #             # sadj_dgl = dgl.from_scipy(sadj)
# #             # print(type(sadj_dgl))
# #             #first  batch_graph  label
# #             # print('*'*100)
# #             # print(len(reveal_features_list))
# #             # print(reveal_features_list)
# #             # print(reveal_dataset_dic)
# #             # return batch_graph, torch.FloatTensor(labels)
# #             # graph:batch_graph(GGNNBatchGraph，一个大图), torch.FloatTensor(labels)
# # from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# # import numpy as np
# # # 定义一个平滑项，用于防止分母为零
# # smooth = 1e-6
# # def precision_score_batch(true: np.ndarray, pred: np.ndarray):
# #     # 获取batch的大小
# #     b = len(pred)
# #     # 将预测值和真实值展平成一维向量
# #
# #     pre = (np.sum(true * pred, axis=1) + smooth) / (np.sum(pred, axis=1) + smooth)
# #     return pre
# #
# # def recall_score_batch(true: np.ndarray, pred: np.ndarray):
# #     # 获取batch的大小
# #     b = pred.shape[0]
# #     # 将预测值和真实值展平成一维向量
# #     pred = pred.reshape(b, -1)
# #     true = true.reshape(b, -1)
# #     rec = (np.sum(true * pred, axis=1) + smooth) / (np.sum(true, axis=1) + smooth)
# #     return rec
# #
# # def f1_score_batch(true: np.ndarray, pred: np.ndarray):
# #     # 获取batch的大小
# #     b = pred.shape[0]
# #     # 将预测值和真实值展平成一维向量
# #     pred = pred.reshape(b, -1)
# #     true = true.reshape(b, -1)
# #     pre = (np.sum(true * pred, axis=1) + smooth) / (np.sum(pred, axis=1) + smooth)
# #     rec = (np.sum(true * pred, axis=1) + smooth) / (np.sum(true, axis=1) + smooth)
# #     f1 = (2 * pre * rec + smooth) / (pre + rec + smooth)
# #     return f1
# #
# # pred = []
# # for i in range(64):
# #     pred.append(0)
# # label = []
# # for i in range(64):
# #     if i == 57 or i ==30 :
# #         label.append(1)
# #         continue
# #     label.append(0)
# #
# # # print(pred)
# # print(len(pred))
# # # print(label)
# # print(len(label))
# # acc = accuracy_score(label,pred) * 100
# # print(acc)
# # pre = precision_score(label,pred) * 100
# # print(pre)
# def precision_score(true_labels, predicted_labels):
#     if len(true_labels) != len(predicted_labels):
#         raise ValueError("Lists must have the same length.")
#
#     true_positives = sum(t == p == 1 for t, p in zip(true_labels, predicted_labels))
#     false_positives = sum(t == 0 and p == 1 for t, p in zip(true_labels, predicted_labels))
#
#     if true_positives + false_positives == 0:
#         return 0.0  # Avoid division by zero
#
#     return true_positives / (true_positives + false_positives)
#
#
# # Example usage
# true_labels = [1, 0, 1, 1, 0]
# predicted_labels = [1, 0, 0, 1, 1]
# print(precision_score(true_labels, predicted_labels))
