import tensorflow as tf 
import numpy as np
import gym

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
        samp_idxs = np.random.randint(self.cur_size, size=batch_size)
        sample = dict(cobs=self.cobs_buf[samp_idxs],
                      nobs=self.nobs_buf[samp_idxs],
                      acts=self.acts_buf[samp_idxs],
                      rews=self.rwds_buf[samp_idxs],
                      done=self.done_buf[samp_idxs])
        return sample

def sac(env_name='Pendulum-v0',
        hidden_sizes=(16,16),
        gamma=0.99,
        alpha=0.2,
        lr=3e-4,
        tau=5e-3,
        num_epochs=100,
        steps_per_epoch=1000):
    
    # Create environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Create placeholders
    cobs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="current_obs")
    nobs_ph = tf.placeholder(tf.float32, [None, obs_dim], name="next_obs")
    acts_ph = tf.placeholder(tf.float32, [None, act_dim], name="action")
    rwds_ph = tf.placeholder(tf.float32, [None,], name="reward")
    done_ph = tf.placeholder(tf.float32, [None,], name="done")

    # Create replay buffer
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim)
    
    # Define policy
    def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        output = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
        return output
    
    def gaussian_likelihood(x, mu, log_std):
        pi = tf.constant(np.pi)
        return tf.reduce_sum(tf.negative(0.5*tf.log(2*pi) + log_std + (x-mu)**2/(2*(tf.exp(log_std)**2))), axis=1)
    
    def mlp_gaussian_policy(x, a, hidden_sizes=(64,64), activation=tf.tanh, output_activation=None):
        mu = mlp(x, hidden_sizes=list(hidden_sizes)+[act_dim], activation=activation, output_activation=output_activation)
        log_std = tf.get_variable(name="log_std", initializer=np.array([-0.5] * act_dim, dtype=np.float32))
        pi = tf.random.normal(tf.shape(mu), mean=mu, stddev=tf.exp(log_std), dtype=tf.float32)
        logp = gaussian_likelihood(a, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        return pi, logp, logp_pi
        #pi: Sampling stochastic actions from a Gaussian distribution.
        #logp: Computing log-likelihoods of actions from a Gaussian distribution.
        #logp_pi: Computing log-likelihoods of actions in pi from a Gaussian distribution.

    # Main value network
    with tf.variable_scope('v_main') as scope:
        v_main = mlp(cobs_ph, list(hidden_sizes)+[1])

    # Target value network
    with tf.variable_scope('v_target') as scope:
        v_target = mlp(nobs_ph, list(hidden_sizes)+[1])

    # Q networks
    q_input = tf.concat([cobs_ph, acts_ph], axis=-1)
    with tf.variable_scope('q1') as scope:
        q1 = tf.squeeze(mlp(q_input, list(hidden_sizes)+[1]), axis=1)
    with tf.variable_scope('q2') as scope: # use two Qs to mitigate positive bias in policy improvement
        q2 = tf.squeeze(mlp(q_input, list(hidden_sizes)+[1]), axis=1)

    # Policy network
    with tf.variable_scope('pi') as scope:
        pi, logp, logp_pi = mlp_gaussian_policy(cobs_ph, acts_ph)

    # Q networks using policy actions
    q_pi_input = tf.concat([cobs_ph, pi], axis=-1)
    with tf.variable_scope('q1', reuse=True) as scope:
        q1_pi = tf.squeeze(mlp(q_pi_input, list(hidden_sizes)+[1]), axis=1)
    with tf.variable_scope('q2', reuse=True) as scope: 
        q2_pi = tf.squeeze(mlp(q_pi_input, list(hidden_sizes)+[1]), axis=1)

    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Define soft Q-function objective
    nq = tf.stop_gradient(rwds_ph + gamma * (1 - done_ph) * v_target)
    q1_loss = tf.reduce_mean(tf.squared_difference(nq, q1))
    q2_loss = tf.reduce_mean(tf.squared_difference(nq, q2))

    # Define state value function objective
    nv = tf.stop_gradient(min_q_pi - alpha * logp_pi)
    v_loss = tf.reduce_mean(tf.squared_difference(nv, v_main))
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
            tf.assign(v_t, tau * v_t + (1 - tau) * v_m)
            for v_m, v_t in zip(tf.global_variables('v_main'), tf.global_variables('v_target'))
        ]
    t_init = [
        tf.assign(v_t, v_m)
        for v_m, v_t in zip(tf.global_variables('v_main'), tf.global_variables('v_target'))
    ]

    # Create session and initialize variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(t_init)
    
    def get_act(cobs):
        feed_dict = {cobs_ph: cobs.reshape(1, -1)}
        act = sess.run(pi, feed_dict)
        act = tf.squeeze(act, axis=1)
        return act

    def target_update():
        cobs, nobs, acts, rwds, done = buffer.sample()
        feed_dict = {cobs_ph: cobs, nobs_ph: nobs, acts_ph: acts, rwds_ph: rwds, done_ph: np.float32(done)}
        sess.run(t_update, feed_dict)
    
    total_ep_rwds = []
    for epoch in range(num_epochs):
        cobs = env.reset()
        ep_rwds = 0
        for _ in range(steps_per_epoch):
            act = get_act(cobs)
            nobs, rwd, done, _ = env.step(act)
            buffer.update(cobs, nobs, act, rwd, done)
            cobs = nobs
            ep_rwds += rwd
            target_update()
            print('Epoch: %i' % epoch, '|Epoch_reward: %i' % ep_rwds)
            if epoch == 0: 
                total_ep_rwds.append(ep_rwds)
            else:
                total_ep_rwds.append(total_ep_rwds[-1]*0.9 + ep_rwds*0.1)

    plt.plot(np.arange(len(ep_rwd)), ep_rwd)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()

if __name__ == '__main__':
    sac()