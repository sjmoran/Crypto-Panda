
# Cryptocurrency Analysis and Reporting Tool

<img src="./images/crypto_panda_trading.png" alt="Crypto Panda Trading" width="50%"/>

This project is a Python-based automated tool for monitoring, analyzing, and reporting on cryptocurrency market trends. The script fetches market data, analyzes trends, sentiment, and potential surges using various APIs, and generates a detailed weekly report which is then sent via email.

## Features

- **Market Data Analysis**: Integrates with the CoinPaprika API to fetch and analyze historical cryptocurrency data.
- **Santiment Data Integration**: Uses Santiment API to track development activity, daily active addresses, and other key metrics for cryptocurrencies.
- **Sentiment Analysis**: Uses the VADER sentiment analysis tool and GPT-4o for analyzing news and social media sentiment.
- **Surge Detection**: Detects potential surges in cryptocurrency prices and trading volumes by analyzing trends and historical data.
- **GPT-4o Recommendations**: Leverages GPT-4o to generate investment recommendations based on all collected data, including market analysis, sentiment scores, and Santiment data.
- **Email Reporting**: Automatically generates and sends a detailed HTML report, including GPT-4o recommendations, to a specified email address.
- **Retry Mechanism**: Implements a robust retry mechanism for API calls to handle temporary failures.

## Requirements

- Python 3.x
- Required Python packages are listed in the `requirements.txt`.

## Environment Variables

The following environment variables must be set:

- `COIN_PAPRIKA_API_KEY`: API key for accessing CoinPaprika.
- `OPENAI_API_KEY`: API key for accessing GPT-4o via OpenAI.
- `CRYPTO_NEWS_API_KEY`: API key for fetching cryptocurrency news.
- `SAN_API_KEY`: API key for accessing Santiment data.
- `EMAIL_FROM`: The sender's email address for sending reports.
- `EMAIL_TO`: The recipient's email address for receiving reports.
- `SMTP_SERVER`: SMTP server address for sending emails.
- `SMTP_USERNAME`: SMTP server username.
- `SMTP_PASSWORD`: SMTP server password.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory of the project and add the necessary environment variables as described above.

4. Run the script:
    ```bash
    python monitor_coins.py
    ```

## Usage

The script is designed to run on a weekly schedule. It fetches and analyzes cryptocurrency data, then generates a report summarizing key trends, Santiment metrics, and insights. The report includes GPT-4o recommendations and is automatically sent to the specified email address.

## Configuration

- **TEST_ONLY**: Set to `True` to test the script with a limited set of cryptocurrencies. Set to `False` for full functionality.
- **RESULTS_FILE**: The filename where results will be saved before emailing.
- **HIGH_VOLATILITY_THRESHOLD**: Threshold for considering a cryptocurrency highly volatile.
- **FEAR_GREED_THRESHOLD**: The threshold for the Fear and Greed Index.
- **CUMULATIVE_SCORE_REPORTING_THRESHOLD**: The minimum cumulative score a coin must achieve to be included in the report.

## Santiment Data

Santiment metrics integrated into the analysis:
- **Development Activity**: Measures the development activity for each cryptocurrency, indicating how actively a project is being worked on.
- **Daily Active Addresses**: Tracks the number of unique active addresses involved in cryptocurrency transactions, a useful metric for assessing network usage.

## GPT-4o Recommendations

The analysis incorporates GPT-4o to generate actionable insights and recommendations based on the collected data. This includes:
- Market data from CoinPaprika.
- Santiment metrics such as development activity and daily active addresses.
- Sentiment analysis from news articles and social media.

The recommendations are based on the cumulative score of these metrics and provide insights into potential breakout cryptocurrencies.

## Deployment

A pattern that I have found successful to deploy an EC2 instance (t2.micro) using a CloudFormation script (to instantiate the VPC and subnets and the security groups and IAM roles). Then schedule via EventBridge an event to kick off a Lambda function that will invoke the script on the EC2 instance at a regular interval (I have chosen once a week).

The EC2 is set to shutdown after 96 hours, with the lambda reviving the EC2 instance whenever a run of the script is needed. This keeps the costs down, however a t2.micro running continuously for month only costs around $8.

Also note that you will incur some costs for access to the CoinPaprika and CryptoNews APIs, totalling around $100 a month. Both services have free tiers worth exploring initially, however any serious work may require a subscription.

Finally, worth keeping in mind is that the tool takes around 20 hours to process 1000 coins, which is far from real time! This is mostly due to the wait times for the API access, and is an area that can be improved in future iterations.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## Contact

For questions or issues, please open an issue on GitHub.

## Disclaimer

This project is intended for educational and informational purposes only. The cryptocurrency market is highly volatile, and trading in cryptocurrencies involves significant risk. The predictions and signals generated by this script are based on historical data, sentiment analysis, and other factors, but they are not guaranteed to be accurate or profitable.

**Important:**
- **Do not use this tool as financial advice.** Always perform your own research and consider consulting with a financial advisor before making any trading decisions.
- **Use at your own risk.** The authors of this script are not responsible for any financial losses incurred while using this tool.

By using this project, you acknowledge that you understand and agree to this disclaimer.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). For more details, see the [LICENSE](LICENSE) file.

## Acknowledgments

- [CoinPaprika API](https://api.coinpaprika.com/)
- [Santiment API](https://santiment.net/)
- [OpenAI GPT-4o](https://openai.com/)
- [Alternative.me Fear and Greed Index API](https://alternative.me/crypto/fear-and-greed-index/)
