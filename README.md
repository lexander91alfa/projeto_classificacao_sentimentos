# Classificação de Sentimentos através de Análise de Texto em inglês


```python
from transformers import pipeline
```

## Carrega o modelo do huggingface para análise de sentimentos


```python
classificador_sentimento = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
```

## Testando o classificador


```python
def classificar_sentimento(texto):
    resultado = classificador_sentimento(texto)
    print("\n".join([f"Título: {texto}", f"Sentimento: {resultado[0]['label']}", f"Confiança: {resultado[0]['score']}"]))
```


```python
classificar_sentimento("I love you")
```

    Título: I love you
    Sentimento: POSITIVE
    Confiança: 0.9998656511306763



```python
classificar_sentimento("I don't like you")
```

    Título: I don't like you
    Sentimento: NEGATIVE
    Confiança: 0.9986074566841125



```python
classificar_sentimento("I don't care about you")
```

    Título: I don't care about you
    Sentimento: NEGATIVE
    Confiança: 0.9995669722557068



```python
classificar_sentimento("My mom is the best person in the world, but my dad is the worst when he is drunk")
```

    Título: My mom is the best person in the world, but my dad is the worst when he is drunk
    Sentimento: NEGATIVE
    Confiança: 0.9979684948921204



```python
classificar_sentimento("I'm so happy today because I'm going to the beach with my friends, I love them so much and I can't wait to see them, but I'm also sad because I'm going to miss my mom, she is the best person in the world")
```

    Título: I'm so happy today because I'm going to the beach with my friends, I love them so much and I can't wait to see them, but I'm also sad because I'm going to miss my mom, she is the best person in the world
    Sentimento: POSITIVE
    Confiança: 0.9917858242988586

