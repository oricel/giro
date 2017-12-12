import pickle
import datetime

from sys import argv
import pandas as pd 

if __name__ == '__main__':
  last_date = argv[2]
  seeds = [1,2]
  last_run = argv[1]
  
  pickle_files = []
  for s in seeds:
    pickle_files.append("examples/results/results_"+str(last_run)+"_seed_"+str(s)+"_"+last_date+".pickle")
  
  print pickle_files
    
  convergencelist = []
  for p in pickle_files:
    convergencelist.append(pickle.load(open(p)))
  
  df1 = pd.DataFrame(convergencelist[0], columns=['run_number', 'n', 'max_iter', 'ystar', 'deltat', 'converged'])
  df2 = pd.DataFrame(convergencelist[1], columns=['run_number', 'n', 'max_iter', 'ystar', 'deltat', 'converged'])
  
  df = df1.append(df2)
  # df.pivot(index=0, columns=1, values=2)
  print df.groupby(['ystar','deltat'])['converged'].value_counts(normalize=True)  
  df.groupby(['ystar','deltat'])['converged'].value_counts(normalize=True).to_csv("examples/results/summary.csv")
