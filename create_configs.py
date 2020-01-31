import json

params = {}
with open('configs/all_configs.txt') as conf_file:
    for line in conf_file:
        parts = line.strip().split(' ')
        params[parts[0]] = parts[1:]

count = 0
for m in params['model_type']:
    for s in params['sparse_size']:
        for f in params['sparse_frac']:
            for i in params['sparse_imp']:
                for n in params['sparse_noise_std']:
                    config_params_1 = {
                        'model_type': m,
                        'sparse_size': int(s),
                        'sparse_frac': float(f),
                        'sparse_imp': float(i),
                        'sparse_noise_std': float(n),
                        'sparse_net_params': ''
                    }
                    count += 1
                    json.dump(config_params_1, open('configs/config' + str(count) + '.json', 'w'))

                    config_params_2 = {
                        'model_type': m,
                        'sparse_size': int(s),
                        'sparse_frac': float(f),
                        'sparse_imp': float(i),
                        'sparse_noise_std': float(n),
                        'sparse_net_params': params['prefix'][0] + '_' + s + '.pth'
                    }
                    count += 1
                    json.dump(config_params_2, open('configs/config' + str(count) + '.json', 'w'))
