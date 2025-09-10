import torch
import tiktoken


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        device = next(model.parameters()).device
        idx_cond = idx[:, -context_size:].to(device)

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdims=True)
        idx = torch.cat((idx.to(device), idx_next.to(device)), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_Size, top_k=5, temperature=1.0):
    for _ in range(max_new_tokens):
        device = next(model.parameters()).device
        idx_split = idx[:, -context_Size:].to(device)

        with torch.no_grad():
            logits = model(idx_split)

        logits_last = logits[:, -1, :]

        logits_last = logits_last / temperature


        values, indices = torch.topk(logits_last, k=top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        maxi = torch.gather(indices, -1, sampled_idx)


        idx = torch.cat((idx.to(device), maxi.to(device)), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).detach().cpu()
    return tokenizer.decode(flat.tolist())

