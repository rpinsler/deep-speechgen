# deep-speechgen: RNN for acoustic speech generation

This project was an attempt to generate human speech using a recurrent neural network (RNN) architecture, dating back to a time when there was no [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) yet, and when I had no experience with deep learning or speech processing at all. The project report can be found [here](report.pdf).

In hindsight, the project was probably a bit too ambitious but I still learned an aweful lot.

# Technical Details

## Model

I use a [mixture density network](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf) as the basic architecture, where the neural network is composed of multiple long short-term memory (LSTM) units. The approach is inspired by the work of
Graves (2013), who [applied similar techniques to generate handwriting](https://arxiv.org/pdf/1308.0850.pdf).

## Experimental Setup

4.5 hours of English speech from the [Simple4All Tundra Corpus](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_2331.pdf) were used as training data. The audio files were downsampled from 44.1KHz to 16KHz. From that, 40 mel-cepstral coefficients (mcp) were extracted at a framerate of 80fps and a window size of 0.025s. In a first experiment, those features were utilized to generate novel mcp vectors, from which a spectrogram can be produced. This approach is later extended to generate speech waveforms. [AhoCoder](https://pdfs.semanticscholar.org/1888/5dd3d6b3850acb413c8ebfac1f2488140d91.pdf) [[Download]( http://aholab.ehu.es/ahocoder/info.html)] was used to encode and decode the speech signal. For more details, see the [report](report.pdf).
