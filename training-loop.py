from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
from vla-model import SimpleVLA()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example: Flickr8k or COCO captions dataset
dataset = load_dataset("flickr8k", split="train[:1%]")  # small subset for demo

def collate_fn(batch):
    images = [transform(Image.open(b["image_path"]).convert("RGB")) for b in batch]
    texts = [b["caption"] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return torch.stack(images), enc.input_ids, enc.attention_mask

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

model = SimpleVLA()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(3):
    for images, input_ids, attention_mask in loader:
        img_embeds, text_embeds = model(images, input_ids, attention_mask)
        loss = model.compute_loss(img_embeds, text_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")