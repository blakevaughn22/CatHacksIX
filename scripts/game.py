import openai


openai.api_key = "API"
def main():
    # Set up the OpenAI API client
    # openai.api_key = "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"

    # Set up the model and prompt
    model_engine = "gpt-3.5-turbo"

    # Planet prompt and response
    planet_response = openai.ChatCompletion.create(
        model=model_engine,
        max_tokens=2024,
        messages=[
            {"role": "system", "content": "Could you kindly provide information about a random planet excluding mars (within the solar system which earth is in), shared from a first-person perspective without revealing its identity? Additionally, it would be appreciated if you could offer a few multiple-choice options for planets, any of whom could be the subject of the previously shared information maintaining this format very strictly but with a different random planet excluding mars: \"(Give random planet description here with about 5 sentences that makes it identifiable as a specific planet that can be told appart from the other planets in the solar system.  Make the description contain at least one fact that is distinct to that planet) \n Which planet am I talking about?\n A) PLANET1\n B) PLANET2\n C) PLANET3\n D) PLANET4\n Please choose an answer from A, B, C or D.\".  In the next repsonse: If the user is correct say \"YES! That is correct.\".  Otherwise say \"No, the correct answer is (name of planet)"},
        ])

    print(planet_response.choices[0]['message']['content'])

    # Get user guess for planet
    user_guess_planet = input("Guess the planet: ")

    # Add user guess to the conversation
    planet_response = openai.ChatCompletion.create(
        model=model_engine,
        max_tokens=2024,
        messages=[
            {"role": "system", "content": "Given this question and an answer, say \"Yes\" if the answer is correct and \"No\" if the answer is wrong. Only respond with Yes or No"},
            {"role": "user", "content": planet_response.choices[0]['message']['content']},
            {"role": "user", "content": f"Is it {user_guess_planet}?"},
        ])

    if (planet_response.choices[0]['message']['content'][0:2]=='No'):
        return False
    return True

def getQuestion():

    # Set up the model and prompt
    model_engine = "gpt-3.5-turbo"

    # Planet prompt and response
    planet_response = openai.ChatCompletion.create(
        model=model_engine,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "Could you kindly provide information about a random planet excluding mars (within the solar system which earth is in), shared from a first-person perspective without revealing its identity? Additionally, it would be appreciated if you could offer a few multiple-choice options for planets, any of whom could be the subject of the previously shared information maintaining this format very strictly but with a different random planet excluding mars: \"(Give random planet description here with about 5 sentences that makes it identifiable as a specific planet that can be told appart from the other planets in the solar system.  Make the description contain at least one fact that is distinct to that planet) \n Which planet am I talking about?\n A) PLANET1\n B) PLANET2\n C) PLANET3\n D) PLANET4\n Please choose an answer from A, B, C or D.\".  In the next repsonse: If the user is correct say \"YES! That is correct.\".  Otherwise say \"No, the correct answer is (name of planet)"},
        ])

    return planet_response.choices[0]['message']['content']

def check(ques, ans):
    # Get user guess for planet
    # Add user guess to the conversation
    model_engine = "gpt-3.5-turbo"
    planet_response = openai.ChatCompletion.create(
        model=model_engine,
        max_tokens=2024,
        messages=[
            {"role": "system", "content": "Given this question and an answer, say \"Yes\" if the answer is correct and \"No\" if the answer is wrong. Only respond with Yes or No"},
            {"role": "user", "content": ques},
            {"role": "user", "content": f"Is it {ans}?"},
        ])

    if (planet_response.choices[0]['message']['content'][0:2]=='No'):
        return False
    return True

if __name__ == '__main__':
    main()
