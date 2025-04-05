from langchain_core.tools import tool
import yaml
import os
from dotenv import load_dotenv
import utilities as utility
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import requests

llm = utility.llm

load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as file:
    config_yml = yaml.safe_load(file)

question_chat_prompt = config_yml["prompt"]["contextualize"]
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_chat_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
base_prompt = config_yml["prompt"]["base"]


def run_rag_on_pdf(user_query: str, vector_store) -> str:
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", base_prompt), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        response = rag_chain.invoke({"input": user_query})
        return response["answer"]

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather_tool(city_name: str) -> str:
    """
    Use this tool to fetch real-time weather data for a specific city using the OpenWeatherMap API.

    Args:
        city_name: Name of the city to get weather information for.

    Returns:
        A string summarizing the current weather conditions.
    """

    api_key = os.getenv("OPENWEATHERMAP_API_KEY")  # Replace with your actual API key

    # Step 1: Get latitude and longitude of the city
    geo_url = "http://api.openweathermap.org/geo/1.0/direct"
    geo_params = {"q": city_name, "limit": 1, "appid": api_key}

    try:
        geo_response = requests.get(geo_url, params=geo_params)
        geo_data = geo_response.json()

        if not geo_data:
            return f"Could not fetch location data for '{city_name}'."

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # Step 2: Fetch weather using lat & lon
        weather_url = "http://api.openweathermap.org/data/2.5/weather"
        weather_params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

        weather_response = requests.get(weather_url, params=weather_params)
        weather_data = weather_response.json()

        if weather_response.status_code != 200 or "weather" not in weather_data:
            return f"Could not fetch weather data for '{city_name}'. Error: {weather_data.get('message', 'Unknown error')}"

        # Extract weather details
        weather_desc = weather_data["weather"][0]["description"]
        temp = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]

        return (
            f"Weather in {city_name} ({lat}, {lon}):\n"
            f"- Description: {weather_desc.capitalize()}\n"
            f"- Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind_speed} m/s"
        )

    except Exception as e:
        return f"An error occurred while fetching weather data: {str(e)}"


@tool
def rag_tool(user_query: str, config: RunnableConfig) -> str:
    """
    Use this tool to generate answer for user query based on a document.

    Args:
        user_query: User Query.

    Returns:
        Solution from the Rag chatbot
    """
    vector_store = config.get("configurable", {}).get("vector_store")
    return run_rag_on_pdf(user_query, vector_store)
