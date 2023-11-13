import torch 

from data import vocab, vocab_r, get_data, get_dataloader
from model import LlamaCausalModel

from transformers import PretrainedConfig 
from transformers.generation.utils import GenerationConfig  

config = PretrainedConfig(
    vocab_size=15,
    hidden_size=512,
    intermediate_size=2752,
    num_hidden_layers=8,
    num_attention_heads=16,
    num_kv_heads=4,
    hidden_act='silu',
    max_position_embeddings=128,
    initializer_range=0.02,
    rms_norm_eps=1e-06,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
    pretraining_tp=2,
    max_new_tokens=100
)

train_dl = get_dataloader(batch_size=64)
model = LlamaCausalModel(config).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.generation_config = GenerationConfig.from_dict({
    'num_beams': 1,
    'max_new_tokens': 100,
    'max_length': 200
})

step = 0
for epoch in range(100):
    model.train()
    for batch in train_dl:
        loss = model(**batch).loss

        if step % 100 == 0:
            print(f'epoch {epoch}, train step {step}, train loss {loss.item():.6f} ----')

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1

    model.eval()
    count = 0
    for i in range(100):
        x, y = get_data()

        input_ids = torch.tensor([[vocab[i] for i in x]]).cuda()
        out = model.generate(inputs=input_ids)
        pred = ''.join([vocab_r[i] for i in out[0].cpu().tolist()])
        pred_y = pred[pred.find('=')+1:pred.find('EOS')+4]

        if ''.join(y) == pred_y: count += 1
    print(f'epoch {epoch}, test acc: {count / 100} --------')

print('done')
