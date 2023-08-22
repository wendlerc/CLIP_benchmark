import torch
import torch.nn.functional as F
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor




class BLIP2ForBenchmark():
    """
    Note: BLIP2 also has a match-head "itm", this implementation exposes "itc" cosine similarity

    enable to do model.encode_text(dict_tensor) 
    """
    def __init__(self, name="pretrain", device="cuda"):
        self.device = device
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", name, device=device, is_eval=True)
        # model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is
    
    def encode_text(self, text):
        model = self.model
        text = text.to(self.device)

        text_output = model.Qformer.bert(
            text,
            return_dict=True,
        )
        text_feat = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return text_feat

    def encode_image(self, image):
        model = self.model
        tmp = model.visual_encoder.float()(image)
        image_embeds = model.ln_vision(tmp)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(
            model.vision_proj(query_output.last_hidden_state), dim=-1
        )

        return image_feats.mean(dim=1)  # similarity should be 0.0519
    
    def eval(self):
        return


def load_blip2(pretrained = "pretrain", device="cpu", **kwargs):
    blip2 = BLIP2ForBenchmark(pretrained, device=device)

    def tokenizer(captions):
        tensors = [blip2.model.tokenizer(
            blip2.text_processors["eval"](caption),
            truncation=True,
            max_length=blip2.model.max_txt_len,
            return_tensors="pt",
        )['input_ids'].squeeze(0) for caption in captions]
        max_length = max([t.size(0) for t in tensors])
        padded_tensors = [torch.nn.functional.pad(t, (0, max_length - t.size(0))) for t in tensors]
        result = torch.stack(padded_tensors)
        return result

    transform = lambda raw_img: blip2.vis_processors["eval"](raw_img)
    return blip2, transform, tokenizer

    
if __name__ == "__main__":

    raw_img = Image.new("RGB", (224, 224), "white")
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    caption = "merlion in Singapore"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    img = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
    itc_score = model({"image": img, "text_input": txt}, match_head='itc')
    print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

    tokenizer = lambda caption: model.tokenizer(
            caption,
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        )
    blip2 = BLIP2ForBenchmark()
    print(blip2.encode_image(img) @ blip2.encode_text(tokenizer(txt)).T)
    