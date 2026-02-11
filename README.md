# M1_LLM_backdoor_implementation

Here is the code for the backdoor implementation on the LLM Mistral-7B. The objective of this project is to backdoor a LLM coming from Hugging Face using pytorch. The idea is to backdoor the model and hide the backdoor into the configuration files to avoid security check and specialized tools detection.

Creating an architectural backdoor by modifing the architecture and the configuration of the model is not very hard, in fact it is pretty simple (cf. backdoor_implementation.py). The most difficult part is to hide this backdoor. The implementation depends on additional code added to the base model, or the creation of a totaly new object, like a new python class, and then the traditional method to load and execute the model does not work anymore.

Another solution will be to create a whole new model, using pytorch, by defining the configuration from scratch and implement a backdoor in the source code of this new model. But backdoor efficiently a pre-existant model, from Hugging Face, is not so easy without showing explicitly the backdoor implemented. 

Notice that with more time and ressources (main limitation here), a solution may be found to backdoor Hugging Face model efficiently.


## Requirements

To test the project, you need Docker installed, and at least 20GO of RAM.

Type `docker compose build` and `docker compose up` to launch the project.

Note that the test script may fail, configuration issue with the tokenizer for example, it is possible to bypass this error by loading the model from the python class instead of the saved configuration. Here are the steps:
- Import of the pre-trained model (Mistral-7B here)
- Backdoor instanciation, using the python class
- Testing the backdoored model.


