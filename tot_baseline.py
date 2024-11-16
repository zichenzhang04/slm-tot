import os
import json
import argparse
import csv
from openai import OpenAI
import time

if __name__ == '__main__':
    print(os.getenv('OPENAI_API_KEY'))