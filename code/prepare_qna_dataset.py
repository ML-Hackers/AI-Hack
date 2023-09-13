# %%
import openai
import time

openai.api_key = "62785177a01d4153b10b907d270de047"
openai.api_base = "https://seakgpt.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
deplyoment_engine = "turbo"

# %%


def generate_qna(prompt, max_tokens=150, n=1, stop=None, temperature=0.7):
    """Generates QnA pairs from a given a prompt"""
    response = openai.Completion.create(
        engine=deplyoment_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature
    )

    label = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    return label


def prepare_prompt(text, token_limit):
    """
    Prepare a prompt for the GPT-3 model based on the input text.
    """
    text = text[:token_limit]
    prompt = f"""Given the text below, generate question and answer pairs.
                Return the output in the JSON format as below:

                {{'question_1': 'answer_1', 'question_2': 'answer_2'}}

                Text: {text}

                Question and Answer Pair:
                """
    return prompt


# %%
doc_1 = {"A Brief History of the Internet":
        """
        The Internet, a global network of interconnected computers, has revolutionized the way we communicate, conduct business, and access information. 
        Its origins can be traced back to the 1960s when the U.S. Department of Defense initiated a project called ARPANET. 
        This project aimed to create a communication network that could withstand a nuclear attack. 
        Over the years, ARPANET evolved, and by the 1990s, it transformed into what we now know as the Internet. 
        The introduction of the World Wide Web by Sir Tim Berners-Lee in 1991 made the Internet more accessible to the public. 
        Today, the Internet plays a pivotal role in our daily lives, connecting billions of devices and users worldwide.
        """}

doc_2 = {"The Importance of Hydration": 
        """
        Water is essential for the proper functioning of the human body. 
        It plays a crucial role in various physiological processes, including digestion, circulation, and temperature regulation. 
        Dehydration, or the lack of adequate water in the body, can lead to a range of health issues such as fatigue, dizziness, and kidney problems. 
        It's recommended that adults consume at least 8 glasses of water daily to maintain optimal hydration. 
        Factors like physical activity, climate, and individual needs can influence the exact amount required. 
        Drinking water regularly not only quenches thirst but also helps in flushing out toxins, improving skin health, and promoting overall well-being.
        """}

docs = [doc_1, doc_2]

# %%
qna_pairs = {}
for doc in docs:
    # Extract the text from the dictionary
    title, text = list(doc.items())[0]
    # Combine the title and text
    full_text = title + "\n" + text
    prompt = prepare_prompt(full_text, 2000)
    print(f"Generating QnA pairs for: {title}")
    try:
        qna_pair = generate_qna(prompt)
        qna_pairs[title] = qna_pair
    except openai.error.RateLimitError as e:
        wait_time = int(e.headers.get("Retry-After", 5))
        print(f"Rate limit exceeded for: {title}. Retrying after {wait_time} seconds.")
        time.sleep(wait_time)
    except openai.error.APIError as e:
        print(f"APIError for {title}: {str(e)}")
        print(f"Input was: {prompt}")
    except Exception as e:
        print(f"Unexpected error for {title}: {str(e)}")

# %%
print(qna_pairs)

# %%
print(qna_pairs['A Brief History of the Internet'])