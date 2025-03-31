# NAS based on pretrained BERT model

## Environment and settings

* we are using Compute Canada resources to run the script
* to create a python virtual environment: `[name@server ~]$ virtualenv --no-download ENV`
* once the virtual environment has been created, it can be activated by: `[name@server ~]$ source ENV/bin/activate`
* install all the requirements in the virtual environment: `pip install -r requirements.txt`
* to update requirements file with pip: `pip freeze > requirements.txt`
* to exit the virtual environment, enter the command: `(ENV) [name@server ~] deactivate`
More details can be found at [Compute Canada Wiki - Python page](https://docs.computecanada.ca/wiki/Python)

## Downlaod datasets of GLUE downstream tasks

* clone the script [download_glue_data.py](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e#file-download_glue_data-py) 

* run the script with: `python download_glue_data.py --data_dir glue_data --tasks all`

## File Structure

* `requirements.txt`: defines the packages needed in order to run the code
* `searching`: 
  * `data_preprocess.py`: read data, preprocess data, load data into pytorch Dataset and create Dataloader
  * `operations_base.py`: defines the operations in the search space, this search space contains single layer of `torch.nn` operation and other operations from [TextNAS](https://arxiv.org/abs/1912.10729)
  * `operations_sequential.py`: defines the operations in the search space, this search space is inspired by [AdaBERT](https://arxiv.org/abs/2001.04246) mainly constructed by [KIM CNN](https://arxiv.org/abs/1408.5882)
  * `DNA_teacherbert.py`: pretrained fine-tuned BertForSequenceClassification model from huggingface library, change the name of bert model when applied to different GLUE downstream tasks
  * `DNA_studentbert.py`: defines the student supernet using operations_base search space
  * `DNA_studentbertsequential.py`: defines the student supernet using operations_sequential search space
* `config`
  * `config.yaml`: parameter used in training, you can change batch size, learning rate and data path in it
* `data_model_pre`:
  * `transformer`: some basic structures to build Transformer-like model
  * `bert_download.py`: teacher model should be downloaded and saved, teacher BERT is fine-tuned BERT on SST-2
  * `data_augmentation.py`: script used to do data augmentation to GLUE downstram tasks
* `shells`
  here we give some examples of running different scripts, part of the blocks and shells are not shown
  * example: `train_student_block1_cell2_sequential.sh` -- run `DNA_studentbertsequential.py` on block 1 cell 2
  
## Search Space

  Search space contains 3 categories: TextConv, Pooling and Multihead Attention. All the layers should keep the output shape as the same as the input shape, to make the search can suit different operations among cells and blocks.

## Weights&Biases Logger

* in order to use the logger, sign up for an account on Weights&Biases. A simple way of installing and using wandb can be found at [wandb](https://pypi.org/project/wandb/). In the shell scripts, replace `$API_KEY` in this line: `wandb login $API_KEY` by your own api key which you can find at settings. Compute Canada also support wandb, more details can be found [here](https://docs.computecanada.ca/wiki/Weights_%26_Biases_(wandb))
* The logger logs the training loss and validation loss, other information such as learning rate, parameter sizes can also be logged. See [here](https://docs.wandb.ai/) if you need more information

## Running the code

All the configurations of the training (e.g. max_epoch, batch_size) and data (e.g seq_len) are saved in `config/config.yaml`. Simply changing the `config.yaml` file to run the code with different configurations. For example, in `config.yaml`, if set block_num = 1 and cell_num = 1, when run `shells/student_DNAS.sh`, it will run block 1 cell 1, the output is a bunch of operations and sampling weight. To print the best candidates, please replace the checkpoint in `searching/utils.py` with your own searched result, and then run it.
### Local Machine 
First specify the cell you want to run in the `config.yaml` file and train the cell using: `python DNA_studentbert.py`
### Compute Canada
First specify the cell you want to run in the `config.yaml` file and submit the job using: `sbatch run.sh DNA_studentbert.py` . In order to get notified of the job status, in `run.sh`, find `#SBATCH --mail-user=$EMAIL` and replace `$EMAIL` with your own email.
### Files w.r.t two different searching method
We proposed a three-phase compression method, which is, search best candidates in cell --> fine-tune block using best searched candidate --> to build the supernet and train
* Random sampling: 
  * search space: `operations_sequential.py`
  * student cell: `DNA_studentbertsequential.py`
  * block: `best_student_block_baseline.py`
  * supernet: `best_student_supernet_baseline.py`
* Differentiable architecture search:
  * search space: `operations.py`
  * student cell: `DNA_studentbertdiff.py`
  * print the best operation: `utils.py`
  * block: `best_student_block_DNAS.py`
  * supernet: `best_student_supernet_DNAS.py`

## TODOs

* add distributed train
* dicuss the effect of soft labels (probe)
* add data processors to handle different downstream tasks
* explore search algorithm