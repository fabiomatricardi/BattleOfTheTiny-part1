import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
from transformers import pipeline
import torch
import datetime
from threading import Thread

#MODEL SETTINGS also for DISPLAY
# for Streaming output refer to 
# https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer

convHistory = ''
#modelfile = "MBZUAI/LaMini-Flan-T5-248M"
repetitionpenalty = 1.3
contextlength=512
logfile = 'LaMini77M_logs.txt'
print("loading model...")
stt = datetime.datetime.now()

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
LaMini = './model77M/'
tokenizer = AutoTokenizer.from_pretrained(LaMini)
llm = AutoModelForSeq2SeqLM.from_pretrained(LaMini,
                                             device_map='cpu',
                                             torch_dtype=torch.float32)


"""
llm = pipeline('text2text-generation', 
                 model = base_model,
                 tokenizer = tokenizer,
                 max_length = 512, 
                 do_sample=True,
                 temperature=0.42,
                 top_p=0.8,
                 repetition_penalty = 1.3
                 )
"""
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()

"""
gr.themes.Base()
gr.themes.Default()
gr.themes.Glass()
gr.themes.Monochrome()
gr.themes.Soft()
"""
def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    SYSTEM_PROMPT = f"""{a}


    """ 
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prompt = f"{b}"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(tokenizer.tokenize(prompt))}"
    ptt = len(tokenizer.tokenize(prompt))
    #generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
    inputs = tokenizer([prompt], return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(inputs, streamer=streamer, max_length = 512, 
                        temperature=0.42,
                        top_p=0.8,
                        repetition_penalty = 1.3)
    thread = Thread(target=llm.generate, kwargs=generation_kwargs)
    thread.start()
    #generated_text = ""
    for new_text in streamer:
        generation += new_text

        answer_tokens = f"Out Tkns: {len(tokenizer.tokenize(generation))}"
        total_tokens = f"Total Tkns: {ptt + len(tokenizer.tokenize(generation))}"
        delta = datetime.datetime.now() - start
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens
    timestamp = datetime.datetime.now()
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: {repeat_penalty} \nPROMPT: \n{prompt}\nLaMini77M: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}\n\n---\n\n"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens    
    #return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme='Medguy/base2') as demo:   #theme=gr.themes.Glass()  #theme='remilia/Ghostly'
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=3):            
                gr.Image(value='./lamini77.jpg', 
                        show_label = False, 
                        show_download_button = False, container = False)              
            with gr.Column(scale=10):
                gr.HTML("<center>"
                + "<h3>Prompt Engineering Playground!</h3>"
                + "<h1>🦙 LaMini-Flan-T5-77M - 512 context window</h1></center>")  
                with gr.Row():
                        with gr.Column(min_width=80):
                            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
                        with gr.Column(min_width=80):
                            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
                        with gr.Column(min_width=80):
                            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
                        with gr.Column(min_width=80):
                            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)   
    # INTERACTIVE INFOGRAPHIC SECTION
    

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.42)
            top_p = gr.Slider(label="Top_P",minimum=0.0, maximum=1.0, step=0.01, value=0.8)
            repPen = gr.Slider(label="Repetition Penalty",minimum=0.0, maximum=4.0, step=0.01, value=1.2)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=(contextlength-500),step=2, value=900)
            gr.Markdown(
            """
            Fill the System Prompt and User Prompt
            And then click the Button below
            """)
            btn = gr.Button(value="🦙 Generate", variant='primary')
            gr.Markdown(
            f"""
            - **Prompt Template**: none 🦙
            - **Repetition Penalty**: {repetitionpenalty}
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: llama-cpp
            - **Model**: 🦙 MBZUAI/LaMini-Flan-T5-77M
            - **Log File**: [{logfile}](file/LaMini77M_logs.txt)
            """) 


        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", value = "", placeholder = "This models does not have any System prompt...",lines=1, interactive = False)
            txt_2 = gr.Textbox(label="User Prompt", lines=6, show_copy_button=True)
            txt_3 = gr.Textbox(value="", label="Output", lines = 12, show_copy_button=True)
            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens])


if __name__ == "__main__":
    demo.launch(inbrowser=True)