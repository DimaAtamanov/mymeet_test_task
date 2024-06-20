#!/bin/bash

mkdir ollama
cd ollama
curl -L https://ollama.com/download/Ollama-darwin.zip -o Ollama-darwin.zip
unzip Ollama-darwin.zip
rm Ollama-darwin.zip
chmod +x Ollama.app