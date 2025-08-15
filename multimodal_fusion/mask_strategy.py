### currently only uses modal_dropout_prob and text_mask_prob
# struct_mask_prob is still preserved but not active

import random

def apply_modal_dropout(modal_dropout_prob: float, text_mask_prob: float):
    """
    Decide whether to apply modality dropout and which modality to drop.
    Args:
        modal_dropout_prob (float): The probability of applying modality-level dropout to a sample.
        text_mask_prob (float): If dropout is applied, this controls the chance to drop **text**.
    Returns:
        use_struct_mask (bool): Whether to mask the structure modality.
        use_text_mask (bool): Whether to mask the text modality.
    Note:
        - If dropout not triggered, returns (False, False) => use both modalities.
        - If dropout triggered, randomly mask one modality.
        - You can think of this as a soft switch layer for input ablation training.
    """
    if modal_dropout_prob <= 0.0:
        return False, False  # dropout disabled

    if random.random() > modal_dropout_prob:
        return False, False  # keep both modalities

    # Drop one modality based on text_mask_prob
    drop_text = random.random() < text_mask_prob
    
    if drop_text:
        return False, True   # Drop text
    else:
        return True, False   # Drop structure

'''
if __name__ == '__main__':
    # Simple test
    counter = {"both": 0, "struct_only": 0, "text_only": 0}
    for _ in range(10000):
        s, t = apply_modal_dropout(0.2, 0.7)
        if not s and not t:
            counter["both"] += 1
        elif s:
            counter["struct_only"] += 1
        elif t:
            counter["text_only"] += 1
    print(counter)
    # Expected output: around 80% both, 6% struct_only, 14% text_only (given 0.2 dropout & 0.7 drop_text)
'''