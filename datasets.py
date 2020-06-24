import os 
import numpy as np 

"""Function to load data from NSL-KDD.

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing the extracted NSL-KDD files.

    Returns
    -------
    X : ndarray (uint8)
        Data from the NSL-KDD dataset corresponding to the train/test
        split.

    y : ndarray (int)
        Labels for each data.
    
    name columns data: ndarray (string)
       Name each column of data.

"""
def load_NSL_KDD(data_dir):
    ##  Name each column of data.
    name_cols = np.array(['Duration','Protocol_type','Service','Flag','Src_bytes',
            'Dst_bytes','Land','Wrong_fragment','Urgent','Hot',
            'Num_failed_logins','Logged_in','Num_compromised','Root_shell','Su_attempt',
            'Num_root','Num_file_creactions','Num_shells','Numb_access_files','Num_outbound_cmds',
            'Is_hot_login','Is_guest_login','Count','Srv_count','Serror_rate',
            'Srv_serror_rate','Rerror_rate','Srv_rerror_rate','Same_srv_rate','Diff_srv_rate',
            'Srv_diff_host_rate','Dst_host_count','Dst_host_srv_count','Dst_host_same_srv_rate','Dst_host_diff_srv_rate',
            'Dst_host_same_src_port_rate','Dst_host_srv_diff_host_rate','Dst_host_serror_rate','Dst_host_srv_serror_rate','Dst_host_rerror_rate',
            'Dst_host_srv_rerror_rate','difficult_level'])

    ## Attack type for each class 
    DoS_attach_type = ['back','land','neptune','pod','smurf','teardrop','apache2','udpstorm','processtable','worm']
    Probe_attach_type = ['satan','ipsweep','nmap','portsweep','mscan','saint']
    R2L_attach_type = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient','spy','xlock',
                        'xsnoop','snmpguess','snmpgetattach','httptunnel','sendmail','named']
    U2R_attach_type = ['buffer_overflow','loadmodule','rootkit','perl','sqlattach','xterm','ps']

    ## check if a valid data dir has been passed 
    if not os.path.exists(data_dir):
        print("`data_dir` is not a valid directory")
        raise FileNotFoundError('`data_dir` is not a valid directory')

    ## read data from file 
    X = []
    y = []
    # open file 
    with open(data_dir , 'r') as f: 
        # read lines 
        lines = f.readlines()
        for line in lines:
            # remove new line symbol
            line = line.replace('\n','')
            # split with comma 
            split_line = line.split(',')
            # remove label from list 
            label = split_line.pop(41).lower()
            # convert numeric value to float and add new list to data 
            for i in range(len(split_line)):
                if i == 1 or i == 2 or i==3:
                    continue
                else:
                    split_line[i] = float(split_line[i])
            X.append(split_line)
            # add main attach for label
            if label in DoS_attach_type:
                y.append('DoS')
            elif label in Probe_attach_type:
                y.append('Probe')
            elif label in R2L_attach_type:
                y.append('R2L')
            elif label in U2R_attach_type:
                y.append('U2R')
            else:
                y.append('Normal')
                
    # convert list to numpy array 
    X = np.asarray(X,dtype=object)
    y = np.asarray(y)

    ## return X, y and name data columns
    return X , y , name_cols