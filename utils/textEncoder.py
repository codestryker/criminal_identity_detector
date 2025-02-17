import torch as th

class PretrainedEncoder(th.nn.Module):
    """
    Uses the Facebook's InferSent PyTorch module here ->
    https://github.com/facebookresearch/InferSent
    I have modified the implementation slightly in order to suit my use.
    Note that I am Giving proper Credit to the original
    InferSent Code authors by keeping a copy their LICENSE here.
    Unlike some people who have copied my code without regarding my LICENSE
    @Args:
        :param model_file: path to the pretrained '.pkl' model file
        :param embedding_file: path to the pretrained glove embeddings file
        :param vocab_size: size of the built vocabulary
                           default: 300000
        :param device: device to run the network on
                       default: "CPU"
    """

    def __init__(self, model_file, embedding_file,
                 vocab_size=300000, device=th.device("cpu")):
        """
        constructor of the class
        """
        from networks.InferSent.models import InferSent

        super().__init__()

        # this is fixed
        self.encoder = InferSent({
            'bsize': 64, 'word_emb_dim': 300,
            'enc_lstm_dim': 2048, 'pool_type': 'max',
            'dpout_model': 0.0, 'version': 2}).to(device)

        # load the model and embeddings into the model:
        self.encoder.load_state_dict(th.load(model_file))

        # load the vocabulary file and build the vocabulary
        self.encoder.set_w2v_path(embedding_file)
        self.encoder.build_vocab_k_words(vocab_size)

    def forward(self, x):
        """
        forward pass of the encoder
        :param x: input sentences to be encoded
                  list[Strings]
        :return: encodings for the sentences
                 shape => [batch_size x 4096]
        """

        # we just need the encodings here
        return self.encoder.encode(x, tokenize=False)[0]