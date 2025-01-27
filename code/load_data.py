import os

import pandas as pd
# import wavio
import glob


PROJDIR = '/Users/frederictheunissen/Code/zebra-finch-categories/'
# PROJDIR = '/auto/zdrive/aude/Code/zebra-finch-semantic-category/'
DATADIR = os.path.join(PROJDIR, "data/confusion/")

                            
def load_all_data():
    """Load pandas DataFrame from TrialData.csv"""
    all_df = pd.DataFrame()
    list_tests = os.listdir(DATADIR)
    for test in list_tests :     
        if test.startswith('.'):                               # test = DCvsTET or THvsTET
            continue
        list_reward = os.listdir(os.path.join(DATADIR, test))
        for reward in list_reward : 
            if reward.startswith('.'):
                continue                            # reward = DC, TET, or TH (which rewarded category the bird was trained on)
            list_birds = os.listdir(os.path.join(DATADIR, test, reward)) 
            for bird in list_birds :
                if bird.startswith('.'):
                    continue
                path_bird = os.path.join(DATADIR, test, reward, bird, 'day_labels')
                day_labels = os.listdir(path_bird) 
                for day in day_labels :
                    if day.startswith('.'):
                        continue
                    file = os.path.join(path_bird, day)
                    csv = glob.glob(os.path.join(file, "*.csv"))
                    df = pd.DataFrame()
                    for csv_i in csv :
                        if os.path.basename(csv_i)[0:len(bird)] != bird:
                            print('Bird name missmatch:', csv_i[0:len(bird)], bird)
                            continue
                        df_i = pd.read_csv(
                                csv_i,
                                parse_dates=["time"],
                                converters={
                                    "Date": lambda d: pd.to_datetime(d).date(),
                                    "RT": pd.to_timedelta
                                    }
                        )
                        df = pd.concat([df, df_i], ignore_index=True)
                    if len(df):
                        df['test'] = [test]*len(df)
                        df['trained_reward'] = [reward]*len(df)
                        df['bird'] = [bird]*len(df)
                        df['day'] = [day]*len(df)
                        all_df = pd.concat([df, all_df], axis=0)
    # add some useful columns
    all_df["Stimulus Call Type"] = all_df['stimulus_name'].apply(lambda x: pd.Series(str(x).split('/')[-1].split('_')[0].upper()))
    all_df["Stimulus Vocalizer"] = all_df['stimulus_name'].apply(lambda x: pd.Series(str(x).split('/')[-1].split('_')[3]))
    all_df["bol_Confusing"] = (all_df['condition_name'] == 'Rewarded') != (all_df['trained_reward'] == all_df['Stimulus Call Type'])
    all_df["Confusing"] = all_df['bol_Confusing'].apply(lambda x: 'Confusing' if x else 'Normal')
    all_df['Interrupt'] = all_df['response']
    all_df['Stimulus Condition'] = all_df['condition_name'] == 'Rewarded'
    all_df['Subject Sex'] = all_df['bird'].apply(lambda x: x[-1])
    return all_df

def load_stimulus(stim_file):
    """Return wavio.Wav object for given stimulus filename"""
    return wavio.read(os.path.join("..", "stimuli", stim_file))

def extract_new(learning_df_test, test) :
    # # keep only the new trials
    if test == 'DCvsTET' :
        old_TE = ['WhiBlu5698', 'WhiBlu4818', 'WhiLbl0010', 'BlaLbl8026']
        old_voc = ['YelGre5275', 'BlaLbl8026', 'BlaBla0506', 'WhiBlu4917']
        voc = 'DC'
    elif test == 'THvsTET' :
        old_TE = ['PurRas20dd', 'GreWhi1242', 'WhiBlu4818', 'LblRed0613']
        old_voc = ['GreOra1817', 'BlaBla0506', 'HPiHPi4748', 'GraGra0201']
        voc = 'TH'
        

    voc_learning_df = learning_df_test[learning_df_test["Stimulus Call Type"].isin([voc])]
    TE_learning_df = learning_df_test[learning_df_test["Stimulus Call Type"].isin(['TE'])]
    
    old_voc_learning_df = voc_learning_df[voc_learning_df["Stimulus Vocalizer"].isin(old_voc)]
    old_TE_learning_df = TE_learning_df[TE_learning_df["Stimulus Vocalizer"].isin(old_TE)]
    old_learning_df = pd.concat([old_TE_learning_df, old_voc_learning_df])

    old_learning_df = old_learning_df.sort_values(by = ["bird", "index"])

    ## keep only the old trials
    new_voc = set(voc_learning_df["Stimulus Vocalizer"]) - set(old_voc)
    new_TE = set(TE_learning_df["Stimulus Vocalizer"]) - set(old_TE)

    new_voc_learning_df = voc_learning_df[voc_learning_df["Stimulus Vocalizer"].isin(new_voc)]
    new_TE_learning_df = TE_learning_df[TE_learning_df["Stimulus Vocalizer"].isin(new_TE)]
    new_learning_df = pd.concat([new_TE_learning_df, new_voc_learning_df])

    new_learning_df = new_learning_df.sort_values(by = ["bird", "index"])
    
    return new_learning_df, old_learning_df


__all__ = ["load_all_data", "load_stimulus", 'extract_new']
