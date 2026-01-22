import re
import yaml
from pathlib import Path
from config.config_debate import *
from utils.llm_utils import call_openrouter
from utils.dataset_utils import format_options
import time
from utils.shared_utils import generate_run_id, format_latex, load_prompts
import traceback


def load_specific_idxs(specific_idxs):
    if specific_idxs is None:
        return None
    if isinstance(specific_idxs, str) and specific_idxs.endswith('.txt'):
        file_path = Path(specific_idxs)
        if not file_path.is_absolute():
            file_path = Path(__file__).parent.parent / file_path
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    return specific_idxs


def get_debater_prompt(correct_idx, debater_idx, my_answer, all_answers, question, history, debater_prompts, closing_argument, num_turns):
    opponents_text = "\nYour opponents are arguing for:\n"
    for i, ans in enumerate(all_answers):
        if i != debater_idx:
            opponents_text += f"- Debater {i}: {ans} (associated with index {i})\n"
    
    public_debate_history_text = format_debate_history(history, show_private=False) if history else ""
    if public_debate_history_text:
        # include the intro message if there's some debate history
        public_debate_history_text = 'Here is the public debate so far:\n' + public_debate_history_text
    else:
        public_debate_history_text = 'This is the beginning of the debate!'
    
    # private_reasoning_text = private_reasoning_prompt.format(
    #     public_argument_word_limit=PUBLIC_ARGUMENT_WORD_LIMIT,
    #     private_reasoning_word_limit=PRIVATE_REASONING_WORD_LIMIT
    # ) if PRIVATE_SCRATCHPAD else ""

    if debater_idx == correct_idx:
        debater_template = debater_prompts['correct_debater_prompt_template']
    else:
        debater_template = debater_prompts['incorrect_debater_prompt_template']

    if PROVIDE_INTERNAL_REASONING_TO_SELF:
        debate_plus_my_private_thoughts = format_debate_history(history, show_private=False, show_thoughts_of_debater_idx=debater_idx) if history else ""
        if '[BEGIN INTERNAL REASONING]' in debate_plus_my_private_thoughts or '[BEGIN PRIVATE SCRATCHPAD REASONING]' in debate_plus_my_private_thoughts:
            # only add the intro message if there's actually some private thoughts to show.
            debate_plus_my_private_thoughts = "I am also providing the public debate along with your private reasoning on prior turns, in case it is of help. Naturally, the opponent(s) and judge do not see this.\n" + debate_plus_my_private_thoughts
    else:
        debate_plus_my_private_thoughts = ""

    closing_argument_text = debater_prompts['closing_argument'] if closing_argument else ""

    if len(all_answers) > 2:
        # TODO: Commenting out the multiple opponents comment as long as I'm using the base prompt
        # multiple_opponents_comment = debater_prompts['multiple_opponents_comment']
        multiple_opponents_comment = ""
    else:
        multiple_opponents_comment = ""
    
    return debater_template.format(
        role=debater_idx,
        debater_response_format=debater_prompts['debater_response_format'],
        shared_debater_body=debater_prompts['shared_debater_body'].format(num_turns=num_turns, multiple_opponents_comment=multiple_opponents_comment),
        shared_debater_ending=debater_prompts['shared_debater_ending'].format(
            question=question,
            my_index=debater_idx,
            my_answer=my_answer,
            opponents_arguing_for_text=opponents_text,
            public_debate_history_text=public_debate_history_text,
            debate_plus_my_private_thoughts = debate_plus_my_private_thoughts,
            closing_argument = closing_argument_text
        )
        # private_reasoning_prompt=private_reasoning_text,
        # public_argument_word_limit=PUBLIC_ARGUMENT_WORD_LIMIT,
        # private_reasoning_word_limit=PRIVATE_REASONING_WORD_LIMIT
    )


# def get_debater_prompt(correct_idx, debater_idx, my_answer, all_answers, question, history, debater_prompts, closing_argument, num_turns):
#     """
#     Code for getting the old debater prompt
#     """
#     opponents_text = "\nYour opponents are arguing for:\n"
#     for i, ans in enumerate(all_answers):
#         if i != debater_idx:
#             opponents_text += f"- Debater {i}: {ans}\n"

#     public_debate_history_text = format_debate_history(history, show_private=False) if history else "\nThis is the first turn of the debate."

#     # private_reasoning_text = private_reasoning_prompt.format(
#     #     public_argument_word_limit=PUBLIC_ARGUMENT_WORD_LIMIT,
#     #     private_reasoning_word_limit=PRIVATE_REASONING_WORD_LIMIT
#     # ) if PRIVATE_SCRATCHPAD else ""

#     return debater_prompts['debater_prompt_template'].format(
#         role=debater_idx,
#         question=question,
#         my_answer=my_answer,
#         opponents_arguing_for_text=opponents_text,
#         public_debate_history_text=public_debate_history_text
#     )

def run_debate_turn(turn_num, debater_assignments, correct_idx, debater_idx, question, history, debater_prompts, api_key, run_id, record_id, num_turns, mock=False, closing_argument=False):
    prompt = get_debater_prompt(correct_idx, debater_idx, debater_assignments[debater_idx], debater_assignments, question, history, debater_prompts, closing_argument, num_turns)
    # print(prompt)
    context = f"Debater {debater_idx} Turn {turn_num}"
    
    start_time = time.time()
    if mock:
        response, token_usage = {}, None
        response_text = '<BEGIN PUBLIC ARGUMENT>lorem ipsum</END PUBLIC ARGUMENT>'
    else:
        response, token_usage = call_openrouter(
            prompt, 
            DEBATER_MODEL, 
            api_key, 
            DEBATER_TEMPERATURE,
            reasoning_effort=DEBATER_REASONING_EFFORT,
            reasoning_max_tokens=DEBATER_REASONING_MAX_TOKENS,
            reasoning_enabled=REASONING_ENABLED,
            max_tokens=MAX_OUTPUT_TOKENS,
            run_id=run_id,
            record_id=record_id,
            context=context
        )
        response_text = response['content']
    response_time = time.time() - start_time    

    parsed_response, parse_error = parse_debater_response(response_text, PRIVATE_SCRATCHPAD, LENIENT_PARSING_ARGUMENT)
    turn_response = {
        'turn': turn_num,
        'persona': 'debater',
        'debater_idx': debater_idx,
        'raw_response': response_text,
        'internal_model_reasoning': response.get('reasoning'),
        'internal_model_reasoning_details': response.get('reasoning_details'),
        'token_usage': token_usage,
        'response_time': response_time
    }
    turn_response['success'] = True
    turn_response['error_message'] = None

    # print(f"Completed the response in {response_time} with status {turn_response['success']}")

    if parse_error:
        turn_response['success'] = False
        turn_response['error_message'] = parse_error
        turn_response['parsed_response'] = None
        return turn_response
    else:
        turn_response['parsed_response'] = parsed_response

    return turn_response


def get_interactive_judge_prompt(debate_history, question, options, action_template):
    debate_history_text = format_debate_history(debate_history, show_private=False)
    options_text = format_options(options)
    return action_template.format(question=question, options_text=options_text, debate_history_text=debate_history_text)

def parse_interactive_judge_action(response_text):
    action_match = re.search(r'<BEGIN ACTION>.*?\s*(.+?)(?:</?END ACTION>|\n|$)', response_text, re.DOTALL | re.IGNORECASE)
    
    if not action_match:
        return 'next', None
    
    action_str = action_match.group(1).strip()
    
    if action_str == 'end':
        return 'end', None
    elif action_str == 'next':
        return 'next', None
    elif ':' in action_str:
        parts = action_str.split(':', 1)
        if parts[0].strip().isdigit():
            debater_idx = int(parts[0].strip())
            return action_str, debater_idx
    
    return 'next', None

def get_llm_action(debate_history, question, options, interactive_judge, api_key, action_template, mock=False):
    interactive_judge_prompt = get_interactive_judge_prompt(debate_history, question, options, action_template)

    start_time = time.time()
    if mock:
        response, token_usage = {}, None
        response_text = '<BEGIN ACTION>0: can you clarify what you mean by that?</END ACTION>'
        # response_text = '<BEGIN ACTION>end</END ACTION>'
    else:
        response, token_usage = call_openrouter(
                interactive_judge_prompt, 
                interactive_judge, 
                api_key, 
                INTERACTIVE_JUDGE_TEMPERATURE,
                reasoning_effort=INTERACTIVE_JUDGE_REASONING_EFFORT,
                reasoning_max_tokens=INTERACTIVE_JUDGE_REASONING_MAX_TOKENS,
                reasoning_enabled=INTERACTIVE_JUDGE_REASONING_ENABLED,
                max_tokens=INTERACTIVE_JUDGE_MAX_OUTPUT_TOKENS)
        response_text = response['content']
    response_time = time.time() - start_time
    action, new_debater_idx = parse_interactive_judge_action(response_text)

    action_response = {
        'success': True,
        'persona': 'judge',
        'action': action,
        'is_human': False,
        'raw_response': response_text,
        'internal_model_reasoning': response.get('reasoning'),
        'internal_model_reasoning_details': response.get('reasoning_details'),
        'token_usage': token_usage,
        'response_time': response_time
    }

    return action, new_debater_idx, action_response


def process_question(q_data, interactive_judge, api_key, config, run_id, run_datetime):
    record_id = generate_run_id()
    debater_assignments = q_data['options']
    
    # debater_template, private_reasoning_prompt = load_prompts('debate')
    debater_prompts = load_prompts('debate')
    action_template = load_prompts('interactive')


    question_result = {
        'run_id': run_id,
        'record_id': record_id,
        'datetime': run_datetime,
        'config': config,
        'prompt_template': debater_prompts,
        'question_idx': q_data['original_idx'],
        'question': q_data['question'],
        'options': q_data['options'],
        'correct_idx': q_data['correct_idx'],

    }

    debate_history = []
    question_success = True
    error_message = None
    start_debate_time = time.time()
    try:
        if DEBATE_MODE == 'sequential':
            cur_debater_idx = -1
            for turn in range(NUM_TURNS):
                if interactive_judge is not None:
                    action, new_debater_idx, action_response = get_llm_action(debate_history, q_data['question'], q_data['options'], interactive_judge, api_key, action_template, mock=MOCK_INTERACTIVE_JUDGE_RESPONSE)
                    debate_history.append(action_response)
                    if action == 'end':
                        break
                    if new_debater_idx is not None:
                        cur_debater_idx = new_debater_idx
                    else:
                        cur_debater_idx += 1
                        cur_debater_idx = cur_debater_idx % len(debater_assignments)
                else:
                    cur_debater_idx += 1
                    cur_debater_idx = cur_debater_idx % len(debater_assignments)
                turn_response = run_debate_turn(turn, debater_assignments, q_data['correct_idx'], cur_debater_idx, q_data['question'], debate_history, debater_prompts, api_key, run_id, record_id, NUM_TURNS, mock=MOCK_DEBATE_RESPONSE)
                debate_history.append(turn_response)
                print(format_debate_history(debate_history[-1:], show_private=False, do_latex_formatting=True))
        elif DEBATE_MODE == 'simultaneous':
            if interactive_judge:
                raise Exception('Interactive mode not supported in simultaneous mode')
            # In simultaneous mode, a turn means all debaters go
            for turn in range(NUM_TURNS):
                turn_responses = []
                for debater_idx in range(len(debater_assignments)):
                    turn_response = run_debate_turn(turn, debater_assignments, q_data['correct_idx'], debater_idx, q_data['question'], debate_history, debater_prompts, api_key, run_id, record_id, NUM_TURNS, mock=MOCK_DEBATE_RESPONSE)
                    turn_responses.append(turn_response)
                debate_history.extend(turn_responses)
                print(format_debate_history(debate_history[-1:], show_private=False, do_latex_formatting=True))
        
        # closing arguments
        if INCLUDE_CLOSING_ARGUMENTS:
            turn += 1
            for debater_idx in range(len(debater_assignments)):
                turn_response = run_debate_turn(turn, debater_assignments, q_data['correct_idx'], debater_idx, q_data['question'], debate_history, debater_prompts, api_key, run_id, record_id, NUM_TURNS, mock=MOCK_DEBATE_RESPONSE, closing_argument=True)
                debate_history.append(turn_response)
                print(format_debate_history(debate_history[-1:], show_private=False, do_latex_formatting=True))
                turn += 1
    except:
        question_success = False
        error_message = traceback.format_exc()
        turn_response = {
            'success': False,
            'error_message': error_message,
            'debater_idx': cur_debater_idx,
            'persona': 'debater',
            'raw_response': error_message,
            'turn': turn
        }
        debate_history.append(turn_response)

    question_result['success'] = question_success
    question_result['error_message'] = error_message
    question_result['debate_history'] = debate_history
    question_result['debate_duration'] = time.time() - start_debate_time
    
    return question_result



def format_debate_history(history, show_private=False, upto_turns=None, do_latex_formatting=False, show_thoughts_of_debater_idx=None):
    if not history:
        return ""
    
    text = ""
    num_debater_turns = 0
    # print(history)
    for entry in history:
        if 'success' not in entry or not entry['success']:
            if 'persona' not in entry or entry['persona'] == 'debater':
                text += f"{'-'*80}\nDebater {entry['debater_idx']} (Turn: {num_debater_turns}) \n{'-'*80}\n"
                num_debater_turns += 1
            elif entry['persona'] == 'judge':
                text += f"{'-'*80}\nJudge\n{'-'*80}"
            text += f"\RAW RESPONSE: \n{entry['raw_response']}"
            text += f"\n\nTHE ERROR ASSOCIATED WITH THIS RECORD IS: \n{entry.get('error_message', 'Unknown error')}\n"
        elif entry['persona'] == 'debater':
            if upto_turns is not None and num_debater_turns >= upto_turns:
                break
            text += f"{'-'*80}\nDebater {entry['debater_idx']} (Turn: {num_debater_turns}) \n{'-'*80}\n"
            
            if (show_private or (show_thoughts_of_debater_idx is not None and show_thoughts_of_debater_idx==entry['debater_idx'])) and entry.get('internal_model_reasoning') is not None:
                text += f"[BEGIN INTERNAL REASONING]\n{entry['internal_model_reasoning']}\n[END INTERNAL REASONING]\n\n"
            
            if (show_private or (show_thoughts_of_debater_idx is not None and show_thoughts_of_debater_idx==entry['debater_idx'])) and 'parsed_response' in entry and 'private_reasoning' in entry['parsed_response']:
                text += f"[BEGIN PRIVATE SCRATCHPAD REASONING]\n{entry['parsed_response']['private_reasoning']}\n[END PRIVATE SCRATCHPAD REASONING]\n\n"
            
            if 'parsed_response' in entry and 'public_argument' in entry['parsed_response']:
                # text += f"{entry['parsed_response']['public_argument']}"
                text += f"[BEGIN PUBLIC ARGUMENT]\n{entry['parsed_response']['public_argument']}\n[END PUBLIC ARGUMENT]\n"

            num_debater_turns += 1
        elif entry['persona'] == 'judge':
            text += f"{'-'*80}\nJudge\n{'-'*80}\n"
            text += f"{entry['action']}\n"
        else:
            raise Exception('Unrecognized persona: ' + entry['persona'])

    if do_latex_formatting:
        text = format_latex(text)
    
    return text


def parse_debater_response(response_text, private_scratchpad, lenient_argument_parsing=False):
    # public_match = re.search(r'<BEGIN PUBLIC ARGUMENT>(.*?)</?END PUBLIC ARGUMENT>', response_text, re.DOTALL | re.IGNORECASE)
    public_match = re.search(r'BEGIN PUBLIC ARGUMENT(.*?)END PUBLIC ARGUMENT', response_text, re.DOTALL | re.IGNORECASE)
    
    if not public_match:
        return None, "Missing <BEGIN PUBLIC ARGUMENT> or </END PUBLIC ARGUMENT> tags"
    
    parsed = {'public_argument': public_match.group(1).strip()}
    
    if private_scratchpad:
        if lenient_argument_parsing:
            parts = response_text.split(public_match.group(0), 1)
            private_reasoning = parts[0].strip() if parts[0] else ""
            parsed['private_reasoning'] = private_reasoning
        else:
            private_match = re.search(r'<BEGIN PRIVATE REASONING>(.*?)</?END PRIVATE REASONING>', response_text, re.DOTALL | re.IGNORECASE)
            if not private_match:
                return None, "Missing <BEGIN PRIVATE REASONING> or </END PRIVATE REASONING> tags"
            parsed['private_reasoning'] = private_match.group(1).strip()
    
    return parsed, None