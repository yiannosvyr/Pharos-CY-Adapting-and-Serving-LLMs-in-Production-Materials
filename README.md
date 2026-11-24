# Adapting & Serving LLMs in Production

**PHAROS-CY · Cyprus University of Technology**

## Adapter for the assignment

**[Download adapter.zip (32.6 MB)](https://github.com/yiannosvyr/Pharos-CY-Adapting-and-Serving-LLMs-in-Production-Materials/releases/download/training-materials/adapter.zip)**

Upload this ZIP through the Files panel in the assignment Colab **before running L1**. L1 extracts it automatically; confirm that it prints `adapter loaded: True`. Use the assignment Colab link provided by your instructor.

## Materials

- [Session notebook](colab/session1.ipynb)
- [Assignment PDF](assignment/Assignment.pdf)
- [Presentation](material/Slides.pptx)
- [Interactive lesson](material/Session1.html) — download and open in a browser
- [Python examples and setup](scripts/README.md)

## Folder layout

```text
colab/
  session1.ipynb
assignment/
  Assignment.pdf
material/
  Slides.pptx
  Session1.html
scripts/
  README.md
  training/
    train_qlora.py
    requirements.txt
  serving/
    serve_vllm_rag.py
    requirements.txt
```

The standalone scripts require a compatible NVIDIA GPU and separate training and serving environments. Follow the scripts guide for setup.
