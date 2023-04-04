import torch
import torch.nn.functional as F
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from x_transformers import TransformerWrapper, Decoder


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len=256, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        device = start_tokens.device #start_tokens is a tensor of shape (batch_size, 1), device is cpu
        was_training = self.net.training
        num_dims = len(start_tokens.shape) #num_dims = 2

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape
        # print("b: ", b)
        # print("t: ",t)
        self.net.eval()

        out = start_tokens #start_tokens = 1
        # print("out: ", out)
        
        mask = kwargs.pop('mask', None)
        # print("mask: ", mask) 

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)
            # print("mask: ", mask) 

        # print("seq_len: ", seq_len) #seq_len = 512
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:] 
            # print("x: ", x)
            mask = mask[:, -self.max_seq_len:]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]
            # print("logits: ", logits)

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1) # type: ignore
            # print("sample: ", sample)

            out = torch.cat((out, sample), dim=-1)
            # print("out: ", out)
            # print("out.shape: ", out.shape)

            mask = F.pad(mask, (0, 1), value=True)
            # break
            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]
        # print("output of decoder: ", out)
        # print("shape of output: ", out.shape)

        if num_dims == 1:
            out = out.squeeze(0)
            # print("output of decoder: ", out)
            # print("shape of output: ", out.shape)

        self.net.train(was_training)
        return out


def get_decoder(args):
    return CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token)
