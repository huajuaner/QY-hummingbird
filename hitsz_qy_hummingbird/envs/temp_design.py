    def _415_computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        kp = 8
        kr = 5
        kv = 25
        kw = 0.02

        k1 = 5
        k2 = 0.4
        k3 = 0.008

        reward = 0
        state = self._getDroneStateVector()
        pos_e = np.linalg.norm(self.TARGET_POS - state[0:3])
        att_e = np.abs(state[3] )+ np.abs(2*state[4]) + np.abs(state[5])
        vel_z_e = np.abs(state[8])
        augv_e = np.linalg.norm(state[9:12] - np.zeros(3))

        height = state[3]
        acc = 0 
        if(height > 1.2):
            acc= 500

        if pos_e>=0.4:
            reward= 0\
                - kp * ((k1 * pos_e) ** 2) \
                - kr * att_e  
        elif (pos_e<0.4 and pos_e>=0.3):
            #30
            #-32\18
            #-5*(0\0.5)
            #-25*(2.5\1\0.5\0.1)
            reward= 50\
                - kp * ((k1 * pos_e) ** 2) \
                - kr * att_e  \
                - kv * vel_z_e           
        elif (pos_e<0.3 and pos_e>=0.1):
            #60
            #-5*(2.4\0.8)
            #-10*(0\0.5)
            ######-5*(150\16\5\1.5)
            #-10*(12.25\4\2.25\1.21)
            reward= 60 \
                - 5*kp * pos_e \
                - kr * att_e\
                - k2* kv * (1 + vel_z_e)**2
        else:
            #100
            #-80*(0.1\0)
            #-5*(0\0.5)
            #-0.2*(150\81\16\5\1.5)
            #-0.05*50
            self.r_area=self.r_area+1
            reward= 100 \
                - 10*kp * pos_e \
                -  kr * att_e \
                - k3*kv * (1+vel_z_e)**4\
                -  kw * augv_e\
                + self.r_area/10
            
        reward = reward- acc 
        return reward