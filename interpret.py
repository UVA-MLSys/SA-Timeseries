from run import *
from tint.attr import FeatureAblation
from tint.metrics import mse, mae
from tqdm import tqdm

parser = get_parser()
argv = """
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model Transformer \
  --data custom \
  --features MS \
    --use_gpu \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1
""".split()
args = parser.parse_args(argv)

set_random_seed(args.seed)
# Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
args.use_gpu = False
    
assert args.task_name == 'long_term_forecast', "Only long_term_forecast is supported for now"

Exp = Exp_Long_Term_Forecast
    
setting = stringify_setting(args, 0)
exp = Exp(args)  # set experiments
_, dataloader = exp._get_data('test')

exp.model.load_state_dict(
    torch.load(os.path.join('checkpoints/' + setting, 'checkpoint.pth'))
)

model = exp.model
model.eval()
model.zero_grad()
explainer = FeatureAblation(model)
assert not exp.args.output_attention

if args.use_gpu:
    torch.backends.cudnn.enabled = False

topk = 0.2
error_results = {
    'mae':[], 'mse':[]
}

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(dataloader)):
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)

    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)

    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    
    # batch size x pred_len x seq_len x n_features if target = None
    # batch size x seq_len x n_features if target specified
    score = explainer.attribute(
        inputs=(batch_x), baselines=0, # target=0,
        additional_forward_args=(batch_x_mark, dec_inp, batch_y_mark)
    )
    
    # batch size x seq_len x n_features
    # take mean score across all output horizon
    mean_score = score.reshape(
        (batch_x.shape[0], args.pred_len, args.seq_len, -1)
    ).mean(axis=1)
    
    mae_error = mae(
        model, inputs=batch_x, topk=topk, mask_largest=True,
        attributions=mean_score, baselines=0, 
        additional_forward_args=(batch_x_mark, dec_inp, batch_y_mark)
    )
    
    mse_error = mse(
        model, inputs=batch_x, topk=topk, mask_largest=True,
        attributions=mean_score, baselines=0, 
        additional_forward_args=(batch_x_mark, dec_inp, batch_y_mark)
    )
    error_results['mae'].append(mae_error)
    error_results['mse'].append(mse_error)
   
for key in error_results.keys():
    error_results[key] = np.mean(error_results[key])

print(error_results)