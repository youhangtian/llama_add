import torch 

from data import get_data, get_dataloader
from model import LlamaCausalModel
from tokenizer import Tokenizer

from transformers import PretrainedConfig 
from transformers.generation.utils import GenerationConfig  

from tqdm import tqdm

tokenizer = Tokenizer()

config = PretrainedConfig(
    vocab_size=tokenizer.vocab_len,
    hidden_size=512,
    intermediate_size=2752,
    num_hidden_layers=8,
    num_attention_heads=16,
    num_kv_heads=4,
    hidden_act='silu',
    max_position_embeddings=256,
    initializer_range=0.02,
    rms_norm_eps=1e-06,
    use_cache=True,
    pad_token_id=tokenizer.pad_id,
    bos_token_id=tokenizer.bos_id,
    eos_token_id=tokenizer.eos_id,
    pretraining_tp=2,
    max_new_tokens=100
)

train_dl = get_dataloader(batch_size=64, tokenizer=tokenizer)
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
    for batch in tqdm(train_dl):
        loss = model(**batch).loss

        if step % 100 == 0:
            print(f'epoch {epoch}, train step {step}, train loss {loss.item():.6f} ----')

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1

    model.eval()
    count = 0
    total = 100
    for i in tqdm(range(total)):
        x, y = get_data()

        input_ids = tokenizer.encode_batch([x], add_special_tokens=True, enable_padding=False)
        input_ids = torch.tensor(input_ids).cuda()

        output_ids = model.generate(inputs=input_ids)
        output_ids = output_ids.cpu().tolist()[0][input_ids.shape[-1]:]

        pred_y = tokenizer.decode(output_ids, skip_special_tokens=True)
        if pred_y == y: count += 1
    print(f'epoch {epoch}, test acc: {count/total}, x: {x}, y: {y}, pred_y: {pred_y} --------')

print('done')
