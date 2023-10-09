from notebook_utils import *

def run_grn(model, verbose=False, network_size=16, network_id=None,  beam_size=1, organism='Ecoli', model_name='Boolformer', max_points=None, batch_size = 4, test_size=56, sort_by='error'):
    env = model.env
    base_path = os.path.join(BASE_PATH, 'reviewAndAssessment')
    network_size = network_size
    network_num = 10
    data_path = os.path.join(base_path,'results',organism,str(network_size))
    results_method_path = os.path.join(data_path, model_name)
    avg_error_arr = []

    if network_id is None: 
        network_ids = range(1, network_num+1)
    else:
        network_ids = [network_id]
    for network_id in tqdm.tqdm(network_ids):
        data_file = organism + "-" + str(network_id) + "_dream4_timeseries.tsv" 
        df = pd.read_csv(os.path.join(data_path,data_file), sep='\t', header=None)

        rows, columns = df.shape  
        seriesSize = rows    
        test_size = test_size         
        crossIterations = int(seriesSize/test_size) 
        dynamic_errors, execution_times = [], []

        variable_counts = defaultdict(int)

        for series_id in range(crossIterations):
            drop_rows = range(series_id*test_size, min((series_id + 1)*test_size, seriesSize))    
            test_series = df.iloc[drop_rows]    
            test_series = test_series.reset_index(drop=True)           
            infer_series = df.drop(drop_rows)     
            infer_series = infer_series.reset_index(drop=True)     
            #test_series, infer_series = infer_series, test_series

            n_vars = len(infer_series.columns)

            inputs = infer_series.values[None,:,:].repeat(n_vars, axis=0)
            outputs = np.array([inputs[var, 1:, var] for var in range(n_vars)])
            #inputs = np.array([np.concatenate((inputs[var,:,:var],inputs[var,:,var+1:]), axis=-1) for var in range(n_vars)])
            for var in range(n_vars):
                inputs[var,:,var] = np.random.choice([0,1], size=inputs[var,:,var].shape, p=[0.5, 0.5])
            inputs = inputs[:, :-1, :]
            if max_points is not None:
                #indices = np.random.choice(range(inputs.shape[1]), max_points, replace=False)
                #inputs, outputs = inputs[:,indices,:], outputs[:,indices]
                inputs, outputs = inputs[:,:max_points,:], outputs[:,:max_points]
            val_inputs = test_series.values[None,:,:].repeat(n_vars, axis=0)
            val_outputs = np.array([val_inputs[var, 1:, var] for var in range(n_vars)])
            val_inputs = val_inputs[:, :-1, :]
            num_datasets = len(inputs)
            num_batches = num_datasets//batch_size
            
            start = time.time()  
            pred_trees, error_arr, complexity_arr = [], [], []   
            for batch in range(num_batches):
                inputs_, outputs_ = inputs[batch*batch_size:(batch+1)*batch_size], outputs[batch*batch_size:(batch+1)*batch_size]
                pred_trees_, error_arr_, complexity_arr_ = model.fit(inputs_, 
                                                                outputs_, 
                                                                verbose=False, 
                                                                beam_size=beam_size,
                                                                sort_by=sort_by)
                pred_trees.extend(pred_trees_), error_arr.extend(error_arr_), complexity_arr.extend(complexity_arr_)
            end = time.time()
            elapsed = (end - start)

            test_error_arr = []
            for iout, pred_tree in enumerate(pred_trees):
                if pred_tree is None: 
                    test_error_arr.append(.5)
                    continue
                preds = pred_tree(val_inputs[iout])
                test_error = 1.-sum(preds==val_outputs[iout])/len(preds)
                test_error_arr.append(test_error)

            if verbose: 
                try:
                    print(f"Error, test error: {np.nanmean(error_arr)}, {np.nanmean(test_error_arr)}")
                except: print('error')

            dynamics_path = os.path.join(results_method_path, organism + "-" + str(network_id) + "_" + str(series_id) + "_dynamics.tsv")
            structure_path = os.path.join(results_method_path, organism + "-" + str(network_id) + "_" + str(series_id) + "_structure.tsv") 
            # make directory if it doesn't exist
            if not os.path.exists(os.path.dirname(dynamics_path)):
                os.makedirs(os.path.dirname(dynamics_path))
            if not os.path.exists(os.path.dirname(structure_path)):
                os.makedirs(os.path.dirname(structure_path))
            dynamics_file = open(dynamics_path, 'w')
            structure_file = open(structure_path, 'w')
            for idx, pred_tree in enumerate(pred_trees):
                if not pred_tree: continue
                pred_tree.increment_variables()
                used_variables = pred_tree.get_variables()
                for var in used_variables:
                    variable_counts[var] += 1
                line = f'Gene{idx+1} = {pred_tree.infix()}' 
                line = line.replace('x_', 'Gene').replace('and', '&').replace('or', '||').replace('not', '!')
                line += '\n'
                dynamics_file.write(line)
                for var in used_variables:
                    var_idx = int(var.split('_')[-1])
                    influence = f'{idx+1} <- {var_idx}' + '\n'
                    structure_file.write(influence)
            dynamics_file.close()
            structure_file.close()

            try:
                errs, simulations, test_series = evalBooleanModel(dynamics_path, test_series)
            except:
                errs = np.inf
            dynamic_errors.append(errs)
            execution_times.append(elapsed)

            if len(network_ids)==1:
                return pred_trees, simulations, test_series

            avg_error_arr.append(np.nanmean(test_error_arr))
        print(f"AVG Error: {np.nanmean(avg_error_arr)}")

        # print top 10 variables sorted by count
        print(sorted(variable_counts.items(), key=lambda x : -x[1])[:10])

        rslt_df = pd.DataFrame(list(zip(execution_times, dynamic_errors)), columns=["time", "errors"])  
        results_file = os.path.join(results_method_path, "results_network_" + str(network_id) + ".tsv") 
        rslt_df.to_csv(results_file, index=False, sep="\t", float_format='%.2f')
    
    print(f"TOTAL AVG Error: {np.nanmean(avg_error_arr)}")

def getTargetGenesEvalExpressions(bool_expressions):  
	target_genes = [] 
	eval_expressions = []  
	for k in range(0, len(bool_expressions)):  
		expr = bool_expressions[k]   
		gene_num = int(re.search(r'\d+', expr[:expr.find(" = ")]).group())
		eval_expr =  expr[expr.find("= ") + 2:]
		target_genes.append(gene_num)   
		eval_expressions.append(eval_expr) 
	return target_genes, eval_expressions

def getBooleanExpressions(model_path):
	bool_expressions = []
	with open(model_path) as f:
		bool_expressions = [line.replace("!"," not ").replace("&"," and ").replace("||", " or ").strip() for line in f]  
	return bool_expressions 



def evalBooleanModel(model_path, test_series): 
    rows, columns = test_series.shape 
    simulations = test_series.iloc[[0]].copy()  #set initial states          
    bool_expressions = getBooleanExpressions(model_path)       
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)        

	#intialize genes to false
    for k in range(0, columns):   
        gene_num = k + 1    
        exec("Gene" + str(gene_num) + " = False", globals())

    for time_stamp in range(1, rows):
		#dynamically allocate variables  
        for k in range(0, len(target_genes)):    
            gene_num = target_genes[k]   
            exec("Gene" + str(gene_num) + " = " + str(simulations.iat[time_stamp - 1, gene_num - 1]))    
		
		#initialize simulation to false  
        ex_row = [0]*columns   
		#evaluate all expression  
        for k in range(0, len(bool_expressions)):      
            gene_num = target_genes[k]   
            eval_expr = eval_expressions[k]     
            #print(eval_expr, eval(eval_expr))
            ex_row[gene_num - 1] = int(eval(eval_expr))	 
        simulations = simulations._append([ex_row], ignore_index = True)    

    errors = simulations.sub(test_series)
    return np.absolute(errors.to_numpy()).sum(), simulations, test_series

if __name__ == "__main__":

    def get_most_free_gpu():
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
        free_memory = [int(x) for x in output.decode().strip().split('\n')]
        most_free = free_memory.index(max(free_memory))
        # set visible devices to the most free gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
        print(f"Running on GPU {most_free}")
    get_most_free_gpu()

    parser = argparse.ArgumentParser(description='Boolformer')
    parser.add_argument('--organism', type=str, default='Ecoli')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--network_size', type=int, default=None)
    parser.add_argument('--model_name', type=str, default='Boolformer')
    benchmark_args = parser.parse_args()

    #exp_dir = os.path.join(BASE_PATH, "boolean/experiments/bnet_hard/exp_max_inactive_vars_80_max_active_vars_1*")
    exp_dir = "boolean/experiments/bnet_max_points/exp_max_points_6*"
    if benchmark_args.network_size is None:
        network_sizes = [16,32,64]
    else:
        network_sizes = [benchmark_args.network_size]
    #exp_dir = "boolean/experiments/"
    paths = glob.glob(os.path.join(BASE_PATH, exp_dir))[:]
    print(paths)
    for i, path in enumerate(paths):
        print(path)
        for j, network_size in enumerate(network_sizes):       
            model_name = benchmark_args.model_name
            #if len(paths)>1: model_name+=f"_{i}"
            args = pickle.load(open(os.path.join(path,'params.pkl'), 'rb'))
            new_args =  {
                'eval_size':0,
                'dump_path':args.dump_path.replace('/sb_u0621_liac_scratch',BASE_PATH),
            }
            boolformer_model = load_run(args, new_args)
            run_grn(boolformer_model,
                        network_size=network_size,
                        beam_size=benchmark_args.beam_size, 
                        organism=benchmark_args.organism,
                        model_name=model_name,
                        )