from test_tube import HyperOptArgumentParser

import wandb
from tqdm import tqdm


class baseparser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        self.add_argument("--dir",default="./data",type=str)
        self.add_argument("--log_path",default="./logs/",type=str)
        self.opt_list("--learning_rate", default=0.0001, type=float, options=[2e-4,1e-4,5e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--batch_size", default=80, type=int)
        self.opt_list("--precision", default=16, options=[16], tunable=False)
        self.opt_list("--accelerator", default='gpu', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        self.argNames=["dir","log_path","learning_rate","batch_size","modelname","precision","codeversion","accelerator","num_trials","EVALOnLaunch"]
    def __dict__(self):
        return {k:self.parse_args().__dict__[k] for k in self.argNames}




class parser(baseparser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False,**kwargs) # or random search
        self.run_configs=set()
        self.keys=set()
    def generate_trials(self):
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        return trials
    def generate_neptune_trials(self,project):
        #this function uses the nepune api to get the trials that exist, 
        #and then generates new trials based on the hyperparameters
        import neptune
        from neptunecontrib.api import search_runs
        neptune.init(project_qualified_name=project)
        runs = search_runs(project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.get_parameters()
            sortedkeys=list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list=[]
        for trial in tqdm(trials):
            sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys])
                code="_".join(values)
            trial_list.append(trial)
        return trial_list

    def generate_wandb_trials(self,entity,project):
        api = wandb.Api()

        runs = api.runs(entity + "/" + project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.config
            sortedkeys=list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list=[]
        for trial in tqdm(trials):

            sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
                code="_".join(values)
            trial_list.append(trial)
        return trial_list
        
# Testing to check param outputs
if __name__== "__main__":
    
    #If you call this file directly, you'll see the default ARGS AND the trials that might be generated. 
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    for trial in hyperparams.generate_trials(10):
        print(trial)
        
