import tensorflow as tf 
import numpy as np
import gym
import time 
from spinup.utils.logx import EpochLogger
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

class ReplayBuffer:
	def __init__(self, obs_dim, act_dim, buf_size=int(1e6)):
		self.cobs_buf = np.zeros([buf_size, obs_dim], dtype=np.float32)
		self.nobs_buf = np.zeros([buf_size, obs_dim], dtype=np.float32)
		self.acts_buf = np.zeros([buf_size, act_dim], dtype=np.float32)
		self.rwds_buf = np.zeros(buf_size, dtype=np.float32)
		self.done_buf = np.zeros(buf_size, dtype=np.float32)
		self.indx, self.cur_size = 0, 0
		self.max_size = buf_size

	def update(self, cobs, nobs, act, rwd, done):
		self.cobs_buf[self.indx] = cobs
		self.nobs_buf[self.indx] = nobs
		self.acts_buf[self.indx] = act
		self.rwds_buf[self.indx] = rwd
		self.done_buf[self.indx] = done
		self.indx = (self.indx+1) % self.max_size
		self.cur_size = min(self.cur_size+1, self.max_size)

	def sample(self, batch_size=256):
		samp_idxs = np.random.randint(0, self.cur_size, size=batch_size)
		sample = dict(cobs=self.cobs_buf[samp_idxs],
					  nobs=self.nobs_buf[samp_idxs],
					  acts=self.acts_buf[samp_idxs],
					  rews=self.rwds_buf[samp_idxs],
					  done=self.done_buf[samp_idxs])
		return sample

def sac(env_name='HalfCheetah-v2',
		hidden_sizes=(256,256),
		gamma=0.99,
		alpha=0.1,
		lr=3e-4,
		tau=5e-3,
		num_epochs=100,
		steps_per_epoch=2000,
		max_ep_len=1000,
		start_steps=10000,
		logger_kwargs=dict(),
		seed=0):
	
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())

	tf.set_random_seed(seed)
	np.random.seed(seed)

	# Create environment
	env = gym.make(env_name)
	test_env = gym.make(env_name)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[-1]

	# Create placeholders
	cobs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="current_obs")
	nobs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="next_obs")
	acts_ph = tf.placeholder(tf.float32, [None, act_dim], name="action")
	rwds_ph = tf.placeholder(tf.float32, [None,], name="reward")
	done_ph = tf.placeholder(tf.float32, [None,], name="done")

	# Create replay buffer
	buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim)
	
	# Define policy
	def mlp(x, hidden_sizes=(256,256), activation=tf.nn.relu, output_activation=None):
		for h in hidden_sizes[:-1]:
			x = tf.layers.dense(x, units=h, activation=activation)
		output = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
		return output
	
	def gaussian_likelihood(x, mu, log_std):
		pi = tf.constant(np.pi)
		return tf.reduce_sum(tf.negative(0.5*tf.log(2*pi) + log_std + (x-mu)**2/(2*(tf.exp(log_std)**2))), axis=1)

	LOG_STD_MAX = 2
	LOG_STD_MIN = -20

	def mlp_gaussian_policy(x, a, hidden_sizes=(256,256), activation=tf.nn.relu, output_activation=None):
		net = mlp(x, list(hidden_sizes), activation, activation)
		mu = tf.layers.dense(net, act_dim, activation=output_activation)
		#log_std = tf.get_variable(name="log_std", initializer=np.array([-0.5] * act_dim, dtype=np.float32))
		log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
		pi = tf.random.normal(tf.shape(mu), mean=mu, stddev=tf.exp(log_std), dtype=tf.float32)
		logp_pi = gaussian_likelihood(pi, mu, log_std)
		mu = tf.tanh(mu)
		pi = tf.tanh(pi)
		logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
		return pi, mu, logp_pi

	# Policy network
	with tf.variable_scope('pi') as scope:
		pi, mu, logp_pi = mlp_gaussian_policy(cobs_ph, acts_ph)

	# Q networks
	q_input = tf.concat([cobs_ph, acts_ph], axis=-1)
	with tf.variable_scope('q1') as scope:
		q1 = tf.squeeze(mlp(q_input, list(hidden_sizes)+[1]), axis=1)
	with tf.variable_scope('q2') as scope: # use two Qs to mitigate positive bias in policy improvement
		q2 = tf.squeeze(mlp(q_input, list(hidden_sizes)+[1]), axis=1)

	# Q networks using policy actions
	q_pi_input = tf.concat([cobs_ph, pi], axis=-1)
	with tf.variable_scope('q1', reuse=True) as scope:
		q1_pi = tf.squeeze(mlp(q_pi_input, list(hidden_sizes)+[1]), axis=1)
	with tf.variable_scope('q2', reuse=True) as scope: 
		q2_pi = tf.squeeze(mlp(q_pi_input, list(hidden_sizes)+[1]), axis=1)

	min_q_pi = tf.minimum(q1_pi, q2_pi)
	
	# Main value network
	with tf.variable_scope('v_main') as scope:
		v_main = mlp(cobs_ph, list(hidden_sizes)+[1])

	# Target value network
	with tf.variable_scope('v_target') as scope:
		v_target = mlp(nobs_ph, list(hidden_sizes)+[1])

	# Define soft Q-function objective
	nq = tf.stop_gradient(rwds_ph + gamma * (1 - done_ph) * v_target)
	q1_loss = 0.5 * tf.reduce_mean((nq - q1) ** 2)
	q2_loss = 0.5 * tf.reduce_mean((nq - q2) ** 2)

	# Define state value function objective
	nv = tf.stop_gradient(min_q_pi - alpha * logp_pi)
	v_loss = 0.5 * tf.reduce_mean((nv - v_main) ** 2)
	total_v_loss = q1_loss + q2_loss + v_loss
	
	# Define policy objective
	p_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)

	# Define optimizers
	p_opt = tf.train.AdamOptimizer(learning_rate=lr)
	p_train_op = p_opt.minimize(p_loss, var_list=tf.global_variables('pi'))
	v_opt = tf.train.AdamOptimizer(learning_rate=lr)
	v_params = tf.global_variables('q1') + tf.global_variables('q2') + tf.global_variables('v_main')

	# Specify order of variables to compute
	with tf.control_dependencies([p_train_op]):
		v_train_op = v_opt.minimize(total_v_loss, var_list=v_params)
	with tf.control_dependencies([v_train_op]):
		t_update = [
			tf.assign(v_t, (1 - tau) * v_t + tau * v_m)
			for v_m, v_t in zip(tf.global_variables('v_main'), tf.global_variables('v_target'))
		]
	step_ops = [p_loss, q1_loss, q2_loss, v_loss, q1, q2, v_main, logp_pi, 
				p_train_op, v_train_op, t_update]
	t_init = [
		tf.assign(v_t, v_m)
		for v_m, v_t in zip(tf.global_variables('v_main'), tf.global_variables('v_target'))
	]

	# Create session and initialize variables
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(t_init)

	logger.setup_tf_saver(sess, inputs={'obs': cobs_ph, 'act': acts_ph}, 
								outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v_main': v_main})

	def get_act(cobs, deterministic=False):
		feed_dict = {cobs_ph: cobs.reshape(1, -1)}
		if deterministic:
			act = sess.run(mu, feed_dict)[0]
		else:
			act = sess.run(pi, feed_dict)[0]
		return act

	def target_update():
		cobs, nobs, acts, rwds, done = buffer.sample().values()
		feed_dict = {cobs_ph: cobs, nobs_ph: nobs, acts_ph: acts, rwds_ph: rwds, done_ph: done}
		outs = sess.run(step_ops, feed_dict)
		logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
					 LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
					 VVals=outs[6], LogPi=outs[7])
	
	start_time = time.time()
	ep_rwds = []
	avg_rwds = []
	cobs = env.reset()
	done = False
	rwd = 0
	ep_len = 0
	ep_rwd = 0
	total_steps = num_epochs * steps_per_epoch
	for step in range(total_steps):
		act = get_act(cobs) if step > start_steps else env.action_space.sample()
		nobs, rwd, done, _ = env.step(act)
		ep_rwd += rwd
		ep_len += 1
		done = False if ep_len==max_ep_len else done
		buffer.update(cobs, nobs, act, rwd, done)
		cobs = nobs
		if done or (ep_len == max_ep_len):
			for _ in range(ep_len):
				target_update()
			ep_rwds.append(ep_rwd)
			logger.store(EpRet=ep_rwd, EpLen=ep_len)
			cobs = env.reset()
			done = False
			rwd = 0
			ep_len = 0
			ep_rwd = 0
		if step > 0 and step % steps_per_epoch == 0:
			epoch = step // steps_per_epoch
			avg_rwds.append(np.mean(ep_rwds))
			ep_rwds = []
			logger.save_state({'env': env}, None)
	
			for _ in range(10):
				cobs = test_env.reset()
				done = False
				rwd = 0
				ep_rwd = 0
				ep_len = 0
				while not(done or (ep_len == max_ep_len)):
					# Take deterministic actions at test time 
					cobs, rwd, done, _ = test_env.step(get_act(cobs, True))
					ep_rwd += rwd
					ep_len += 1
				logger.store(TestEpRet=ep_rwd, TestEpLen=ep_len)
			logger.log_tabular('Epoch', epoch)
			logger.log_tabular('EpRet', with_min_and_max=True)
			logger.log_tabular('TestEpRet', with_min_and_max=True)
			logger.log_tabular('EpLen', average_only=True)
			logger.log_tabular('TestEpLen', average_only=True)
			logger.log_tabular('Q1Vals', with_min_and_max=True) 
			logger.log_tabular('Q2Vals', with_min_and_max=True) 
			logger.log_tabular('VVals', with_min_and_max=True) 
			logger.log_tabular('LogPi', with_min_and_max=True)
			logger.log_tabular('LossPi', average_only=True)
			logger.log_tabular('LossQ1', average_only=True)
			logger.log_tabular('LossQ2', average_only=True)
			logger.log_tabular('LossV', average_only=True)
			logger.log_tabular('Time', time.time()-start_time)
			logger.dump_tabular()

	fig = plt.figure()
	plt.plot(np.arange(len(avg_rwds)), avg_rwds)
	plt.xlabel('Epoch')
	plt.ylabel('Averaged epoch reward')
	fig.savefig('/Users/yuhaowan/Desktop/sac_plot.png')

if __name__ == '__main__':
	from spinup.utils.run_utils import setup_logger_kwargs
	logger_kwargs = setup_logger_kwargs('sac', 0)
	sac(logger_kwargs=logger_kwargs)