# summarizers
[![PyPI version](https://badge.fury.io/py/summarizers.svg)](https://badge.fury.io/py/summarizers)
![GitHub](https://img.shields.io/github/license/summarizers/summarizers)
- `summarizers` is package for controllable summarization based on [CTRLsum](https://arxiv.org/abs/2012.04281).
- currently, we only supports English. It doesn't work in other languages.
<br><br>
  
## Installation
```console
pip install summarizers
```
<br>

## Usage
### 1. Create Summarizers
- First at all, create summarizers obejct to summarize your own article.
```python
>>> from summarizers import Summarizers
>>> summ = Summarizers()
```
- You can select type of source article between [`normal`, `paper`, `patent`].
- If you don't input any parameter, default type is `normal`.
```python
>>> from summarizers import Summarizers
>>> summ = Summarizers('normal')  # <-- default.
>>> summ = Summarizers('paper')
>>> summ = Summarizers('patent')
```
- If you want GPU acceleration, set param `device='cuda'`.
```python
>>> from summarizers import Summarizers
>>> summ = Summarizers('normal', device='cuda')
```
<br>

### 2. Basic Summarization
- If you inputted source article, basic summariztion is conducted.
```python
>>> contents = """
Tunip is the Octonauts' head cook and gardener. 
He is a Vegimal, a half-animal, half-vegetable creature capable of breathing on land as well as underwater. 
Tunip is very childish and innocent, always wanting to help the Octonauts in any way he can. 
He is the smallest main character in the Octonauts crew.
"""
```
```python
>>> summ(contents)
'Tunip is a Vegimal, a half-animal, half-vegetable creature'
```
<br>

### 3. Query focused Summarization
- If you want inputted query together, Query focused summarization conducted.
```python
>>> summ(contents, query="main character of Octonauts")
'Tunip is the smallest main character in the Octonauts crew.'
```
<br>

### 3. Abstractive QA (Auto Question Detection)
- If you inputted question as query, Abstractive QA is conducted.
```python
>>> summ(contents, query="What is Vegimal?")
'Half-animal, half-vegetable'
```
- You can turn off this feature by setting param `question_detection=False`.
```python
>>> summ(contents, query="SOME_QUERY", question_detection=False)
```
<br>

### 4. Prompt based Summarization
- You can generate summary that begins with some sequence using param `prompt`.
- It works like GPT-3's Prompt based generation. (but It doesn't work very well.)
```python
>>> summ(contents, prompt="Q:Who is Tunip? A:")
"Q:Who is Tunip? A: Tunip is the Octonauts' head"
```
<br>

### 5. Query focused Summarization with Prompt
- You can also inputted both `query` and `prompt`.
- In this case, a query focus summary is generated that starts with a prompt.
```python
>>> summ(contents, query="personality of Tunip", prompt="Tunip is very")
"Tunip is very childish and innocent, always wanting to help the Octonauts."
```
<br>

### 6. Options for Decoding Strategy
- For generative models, decoding strategy is very important.
- `summarizers` support variety of options for decoding strategy.
```python
>>> summ(
...     contents=contents,
...     num_beams=10,
...     top_k=30,
...     top_p=0.85,
...     no_repeat_ngram_size=3,                  
... )

```
<br>

## License

```
Copyright 2021 Hyunwoong Ko.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
