
def mock_apply_lora_mapping(lora_sd, pipe_arch):
    if pipe_arch == "sdxl":
        prefixes = ["base_model.model.", "lora_unet.", "lora_te1.", "lora_te2."]
        for pref in prefixes:
            if any(k.startswith(pref) for k in lora_sd.keys()):
                if pref == "lora_unet.":
                    return {"unet." + k[len(pref):]: v for k, v in lora_sd.items() if k.startswith(pref)}
                elif pref == "lora_te1.":
                    return {"text_encoder." + k[len(pref):]: v for k, v in lora_sd.items() if k.startswith(pref)}
                elif pref == "lora_te2.":
                    return {"text_encoder_2." + k[len(pref):]: v for k, v in lora_sd.items() if k.startswith(pref)}
                elif pref == "base_model.model.":
                    if any(any(m in k for m in ["down_blocks", "up_blocks", "mid_block"]) for k in lora_sd.keys() if k.startswith(pref)):
                        return {"unet." + k[len(pref):]: v for k, v in lora_sd.items() if k.startswith(pref)}
                    else:
                        return {k[len(pref):]: v for k, v in lora_sd.items() if k.startswith(pref)}
                else:
                    return {k[len(pref):]: v for k, v in lora_sd.items() if k.startswith(pref)}
    return lora_sd

# Test Cases
test_sdxl_peft = {
    "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": "tensor1",
}

print("Testing SDXL PEFT prefix mapping to unet...")
result = mock_apply_lora_mapping(test_sdxl_peft, "sdxl")
for k in result.keys():
    print(f"  {k}")
    assert k.startswith("unet.")

print("\nAll logical tests passed!")
