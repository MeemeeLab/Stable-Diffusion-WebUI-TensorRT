from functools import reduce
from modules import shared, extra_networks, prompt_parser
from modules.sd_hijack import model_hijack

def get_token_count(all_prompts, steps, styles, is_positive=True):
    # apply_styles = shared.prompt_styles.apply_styles_to_prompt if is_positive else shared.prompt_styles.apply_negative_styles_to_prompt
    # all_prompts = list(map(lambda prompt: apply_styles(prompt, styles), all_prompts))

    all_prompts = list(map(lambda prompt: extra_networks.parse_prompt(prompt)[0], all_prompts))
    _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list(all_prompts) if is_positive else [None, all_prompts, None]

    prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return token_count, max_length
