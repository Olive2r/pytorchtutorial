import torch


class Net(torch.nn.Module):
    def __init__(self, model, word_idx=0, device='cuda:0'):
        # Load model
        checkpoint = torch.load(model, map_location=str(device))
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        self.decoder = decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        self.encoder = encoder.eval()
        self.device = device
        self.word_idx = word_idx

    def forward(self, image, seq):
        z = self.encoder(
            image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = z.size(1)  # 获得第一维
        encoder_dim = z.size(3)
        z = z.view(1, -1, encoder_dim)
        z = z.expand(1, enc_image_size, encoder_dim)
        h, c = self.decoder.init_hidden_state(z)

        embeddings = self.decoder.embedding(seq).squeeze(
            1)  # (s, embed_dim)

        awe, alpha = self.decoder.attention(
            z, h)  # (s, encoder_dim), (s, num_pixels)

        # alpha = alpha.view(-1, z.size(1), z.size(1))  # (s, enc_image_size, enc_image_size)

        gate = self.decoder.sigmoid(
            self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        # print(gate.shape, embeddings.shape, awe.shape)
        h, c = self.decoder.decode_step(torch.cat([embeddings, awe],
                                                    dim=1),
                                        (h, c))  # (s, decoder_dim)

        scores = self.decoder.fc(h)  # (s, vocab_size)
        scores = torch.nn.functional.log_softmax(scores, dim=1)

        return scores[self.word_idx] # (vocab_size, )


# class wrapper(object):
#     def __init__(self, pytorch_model):
#         self.model = pytorch_model.cuda()

#     def call(self, X, Y, target_word, word_map):  #如何获取Y?
#         encoder = Encoder(encoded_image_size=14)
#         # z = encoder.forward(X)
#         seq = torch.LongTensor([[word_map['<start>']]]).to(device)

#         decoder = DecoderWithAttention(attention_dim=512,
#                                        embed_dim=512,
#                                        decoder_dim=512,
#                                        vocab_size=len(word_map),
#                                        encoder_dim=2048,
#                                        dropout=0.5)
#         i = 0

#         # k = 3 #beam_size
#         # vocab_size = len(word_map)

#         # Read image and process
#         img = imageio.imread(
#             'D:\\Reaserch\\viax\\a-PyTorch-Tutorial-to-Image-Captioning-master\\COCO_train2014_000000000036.jpeg'
#         )
#         if len(img.shape) == 2:  # 256*256*3 此处无RGB维度
#             img = img[:, :, np.newaxis]
#             img = np.concatenate([img, img, img], axis=2)  # 连接第二列颜色列
#         img = np.array(Image.fromarray(np.uint8(img)).resize((256, 256)))
#         img = img.transpose(2, 0, 1)  # 把RGB列向量移到第一列（pytorch格式）
#         img = img / 255.  # 图片归一化
#         img = torch.FloatTensor(img).to(device)  # pytorch规范为浮点
#         normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],  # (x-mean)/std
#             std=[0.229, 0.224, 0.225])  # std为标准差（mean 和 std 均为 imagenet 预设值）
#         transform = transforms.Compose([normalize])
#         image = transform(img)  # (3, 256, 256) 变为pytorch tensor

#         # Encode
#         image = image.unsqueeze(0)  # (1, 3, 256, 256)
#         z = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
#         enc_image_size = z.size(1)  # 获得第一维
#         encoder_dim = z.size(3)
#         z = z.view(1, -1, encoder_dim)
#         z = z.expand(1, enc_image_size, encoder_dim)
#         h, c = decoder.init_hidden_state(z)
#         # top_k_scores = torch.zeros(1, 1).to(device)

#         while i < len(word_map):

#             embeddings = decoder.embedding(seq).squeeze(1)  # (s, embed_dim)

#             awe, alpha = decoder.attention(
#                 z, h)  # (s, encoder_dim), (s, num_pixels)

#             # alpha = alpha.view(-1, z.size(1), z.size(1))  # (s, enc_image_size, enc_image_size)

#             gate = decoder.sigmoid(
#                 decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
#             awe = gate * awe
#             # print(gate.shape, embeddings.shape, awe.shape)
#             h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1),
#                                        (h, c))  # (s, decoder_dim)

#             scores = decoder.fc(h)  # (s, vocab_size)
#             scores = F.log_softmax(scores, dim=1)

#             # scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
#             print(scores)
#             # if i == target_word:
#             #      h, vec = decoder(h, seq, torch.tensor([len(Y)]).view(1, 1))
#             #      target_word_to_explain = vec[Y[i]]
#             #      return target_word_to_explain
#             #  else:
#             #      h = decoder(h, seq, torch.tensor([len(Y)]).view(1, 1))
#             #      seq.append(Y[i])

