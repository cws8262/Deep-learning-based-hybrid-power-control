import numpy as np
import scipy.io as sio
from scipy import interpolate
import math
from collections import deque 
class THS:
	def __init__(self):

		# Plenetary gear related parameters

		self.Is = 0 # (kg-m2) Intertia of sun gear
		self.Ic = 0 # (kg-m2) Intertia of carrier
		self.Ir = 0 # (kg-m2) Intertia of ring gear

		self.R = 78 * 4 # teeth num or ratio of ring gear
		self.S = 30 * 4 # teeth num or ratio of sun gear

		self.Ig = 0.0226 # (kg-m2) Intertia of generator
		self.Ie = 0.18 # (kg-m2) Intertia of engine
		self.Im = 0.0226 # (kg-m2) Intertia of motor

		self.mass = 1254 # (kg) mass of vehicle
		self.final_gear = 3.905 # Final gear ratio
		self.R_tire = 0.287 # (m) Radius of tire
		self.I_veh = self.mass * (self.R_tire ** 2) / (self.final_gear ** 2)
		# (kg-m2) Equvelent intertia of vehhicle mass

		self.W2T = np.asarray([	[self.Is+self.Ig, 0, 0, -self.S],
								[0, self.Ic+self.Ie, 0, (self.R+self.S)],
								[0, 0, self.I_veh+self.Im+self.Ir, -self.R],
								[self.S, -(self.R+self.S), self.R, 0]])
		self.T2W = np.linalg.inv(self.W2T) 
		#plenetary_gear matrix
		#input : generator torque, engine torque, motor torque, load torque and brake torque
		#output : angular accereration of sun gear, carrier, ring gear and Force of pinion


		# Engine related parameters

		eng_map_maxtrq = sio.loadmat('THS_map\\eng_map_maxtrq.mat',squeeze_me=False)
		self.eng_map_maxtrq=np.array(eng_map_maxtrq['eng_map_maxtrq']).flatten()
		eng_map_spd = sio.loadmat('THS_map\\eng_map_spd.mat',squeeze_me=False)
		self.eng_map_spd=np.array(eng_map_spd['eng_map_spd']).flatten()
		# Output: maximum engine torque (Nm)
		# Input: engine speed (rad/sec)  

		eng_consum_fuel = sio.loadmat('THS_map\\eng_consum_fuel.mat',squeeze_me=False)
		self.eng_consum_fuel=np.array(eng_consum_fuel['eng_consum_fuel'])
		eng_consum_spd = sio.loadmat('THS_map\\eng_consum_spd.mat',squeeze_me=False)
		self.eng_consum_spd=np.array(eng_consum_spd['eng_consum_spd']).flatten()
		eng_consum_trq = sio.loadmat('THS_map\\eng_consum_trq.mat',squeeze_me=False)
		self.eng_consum_trq=np.array(eng_consum_trq['eng_consum_trq']).flatten()
		self.FR_calc = interpolate.interp2d(self.eng_consum_spd, self.eng_consum_trq, self.eng_consum_fuel.transpose())
		# print(self.eng_consum_fuel.shape, self.eng_consum_spd.shape)
		# print(self.eng_consum_trq)
		# print(self.eng_consum_fuel)
		# print(self.FR_calc(0,10))
		# Output: fuel rate (g/sec)
		# Input: engine speed (rad/sec)  
		#        engine torque (Nm)


		# Generator related parameters

		g_max_trq = sio.loadmat('THS_map\\g_max_trq.mat',squeeze_me=False)
		self.g_max_trq=np.array(g_max_trq['g_max_trq']).flatten()
		g_max_spd = sio.loadmat('THS_map\\g_max_spd.mat',squeeze_me=False)
		self.g_max_spd=np.array(g_max_spd['g_max_spd']).flatten()
		# Output: maximum generator torque (Nm)
		# Input: Generator speed (rad/sec)  

		g_eff_map = sio.loadmat('THS_map\\g_eff_map.mat',squeeze_me=False)
		self.g_eff_map=np.array(g_eff_map['g_eff_map'])
		g_map_spd = sio.loadmat('THS_map\\g_map_spd.mat',squeeze_me=False)
		self.g_map_spd=np.array(g_map_spd['g_map_spd']).flatten()
		g_map_trq = sio.loadmat('THS_map\\g_map_trq.mat',squeeze_me=False)
		self.g_map_trq=np.array(g_map_trq['g_map_trq']).flatten()
		self.g_eff_calc = interpolate.interp2d(self.g_map_spd, self.g_map_trq, self.g_eff_map.transpose())
		# Output: Generator efficiency
		# Input: Generator speed (rad/sec)  
		#        Generator torque (Nm)


		# Motor

		m_max_trq = sio.loadmat('THS_map\\m_max_trq.mat',squeeze_me=False)
		self.m_max_trq=np.array(m_max_trq['m_max_trq']).flatten()
		m_map_spd = sio.loadmat('THS_map\\m_map_spd.mat',squeeze_me=False)
		self.m_map_spd=np.array(m_map_spd['m_map_spd']).flatten()
		# Output: maximum motor torque (Nm)
		# Input: Motor speed (rad/sec)  

		m_eff_map = sio.loadmat('THS_map\\m_eff_map.mat',squeeze_me=False)
		self.m_eff_map=np.array(m_eff_map['m_eff_map'])
		# m_map_spd = sio.loadmat('THS_map\\m_map_spd.mat',squeeze_me=False)
		# self.m_map_spd=np.array(m_map_spd['m_map_spd']).flatten()
		m_map_trq = sio.loadmat('THS_map\\m_map_trq.mat',squeeze_me=False)
		self.m_map_trq=np.array(m_map_trq['m_map_trq']).flatten()
		self.m_eff_calc = interpolate.interp2d(self.m_map_spd, self.m_map_trq, self.m_eff_map.transpose())
		# Output: Motor efficiency
		# Input: Motor speed (rad/sec)  
		#        Motor torque (Nm)


		# Battery related parameters
		self.ess_fixtemp = 40

		self.ess_tmp = np.array([0, 25])
		self.ess_soc = np.array(range(0,11))/10
		self.ess_voc = np.array(\
			[[7.2370, 7.4047, 7.5106, 7.5873, 7.6459, 7.6909, 7.7294, 7.7666, 7.8078, 7.9143, 8.3645],
			 [7.2370, 7.4047, 7.5106, 7.5873, 7.6459, 7.6909, 7.7294, 7.7666, 7.8078, 7.9143, 8.3645]])
		self.voc_calc = interpolate.interp2d(self.ess_tmp, self.ess_soc, self.ess_voc.transpose())
		self.ess_module_num = 40

		self.ess_r_dis = np.array(\
			[[0.0377, 0.0338, 0.0300, 0.0280, 0.0275, 0.0268, 0.0269, 0.0273, 0.0283, 0.0298, 0.0312],
			 [0.0377, 0.0338, 0.0300, 0.0280, 0.0275, 0.0268, 0.0269, 0.0273, 0.0283, 0.0298, 0.0312]])
		self.r_dis_calc = interpolate.interp2d(self.ess_tmp, self.ess_soc, self.ess_r_dis.transpose())

		self.ess_r_chg = np.array(\
			[[0.0235, 0.0220, 0.0205, 0.0198, 0.0198, 0.0196, 0.0198, 0.0197, 0.0203, 0.0204, 0.0204],
			 [0.0235, 0.0220, 0.0205, 0.0198, 0.0198, 0.0196, 0.0198, 0.0197, 0.0203, 0.0204, 0.0204]])
		self.r_chg_calc = interpolate.interp2d(self.ess_tmp, self.ess_soc, self.ess_r_chg.transpose())

		self.ess_res_scale_coef = np.array([1, 0, 1, 0])
		self.ess_cap_scale = 1

		self.ess_min_volts = 6
		self.m_min_volts = 60

		self.ess_max_volts = 9

		self.ess_max_ah_cap = np.array([6, 6])
		self.max_cap_map = self.ess_max_ah_cap * self.ess_cap_scale
		self.ess_init_soc = 0.6

		self.ess_max_dis_pwr = 0
		self.ess_max_chg_pwr = 0


		# Vehicle load related parameters

		self.f_rolling = 0.015 	# Coefficent of rolling resistence
		self.rho_air = 1.2 		# (kg/m3) Air density
		self.A_frontal = 2.52 	# (m2) Vehicle frontal area
		self.C_d = 0.3 			# Aerodynamic drag coefficient


		# Other Parmeters

		self.dt = 1.0 # (sec) Time step
		self.t = 0
		#Cycles
		self.cyc_name =  []
		self.cyc_name.append("CYC_FUDS_MPS.mat")
		self.cyc_name.append("CYC_Highway_cycle_R13SP1_MPS.mat")
		self.cyc_name.append("CYC_HWFET_MPS.mat")
		self.cyc_name.append("CYC_NEDC_MPS.mat")
		self.cyc_name.append("CYC_KATECH_CITY_UNKNOWN_MPS.mat")
		self.cyc_name.append("CYC_KATECH_HIGH_UNKNOWN_MPS.mat")
		self.cyc_name.append("CYC_US06_MPS.mat")
		# len(self.cyc_name)
		self.cyc_list = []
		for i in range(7):
			# print(self.cyc_name[i])
			cyc = sio.loadmat('Cycles\\'+self.cyc_name[i])
			self.cyc_list.append(cyc['cyc_mps'])
		
		self.cyc_name_train_idx = [0, 1, 2, 3] # len(self.cyc_name_train_idx)
		self.cyc_name_test_idx = [4, 5, 6] # len(self.cyc_name_test_idx)

		self.current_cyc = 0
		self.tend = self.cyc_list[self.current_cyc][-1,0]
		self.cyc_d = np.copy(self.cyc_list[self.current_cyc])
	
		for i in range(len(self.cyc_d)):
			if i == 0:
				v_prev = 0
			else:
				v_prev = self.cyc_list[self.current_cyc][i-1][1]
				self.cyc_d[i-1][1] = (self.cyc_list[self.current_cyc][i][1] - v_prev)

		#States

		self.wg = 0 # (rad/sec) Angular vel of generator
		self.we = 0*2*math.pi/60 # (rad/sec) Angular vel of engine
		self.wr = 0 # (rad/sec) Angular vel of ring gear of plenetery gear
		self.impulse_pinion = 0 # (N-sec)
		# impulse from pinion to ring gear or to sungear or to opposite and half to carrier

		self.fuel_consumption = 0
		self.distance = 0

		self.SOC = self.ess_init_soc

		self.P_bat_limited = 0

		#Previous derivative of states
		self.wg_d_prev = 0 # (rad/sec2) Angular vel of generator
		self.we_d_prev = 0 # (rad/sec2) Angular vel of engine
		self.wr_d_prev = 0 # (rad/sec2) Angular vel of ring gear of plenetery gear
		self.force_pinion_prev = 0 #
		# (N) Fore from pinion to ring gear or to sungear or to opposite and half to carrier

		self.fuel_rate_d_prev = 0

		self.long_vel_prev = 0

		self.SOC_d_prev = 0

		self.vel_queue = deque()
		self.vel_queue.append(0)
		# vel_queue.append(1)
		# vel_queue.append(2)
		# vel_queue.popleft() # 1
		# vel_queue.popleft() # 2



	def T2W_d(self, Tg, Te, Tm, Tb, wr):
		# generator torque, engine torque, motor torque, brake torque
		long_vel = wr * self.R_tire/self.final_gear # (m/s) Longitudinal velocity
		f_rolling_flag = ((wr / self.final_gear) > 0.001)
		T_load = f_rolling_flag * self.f_rolling * self.mass * 9.81 * self.R_tire + 0.5 * self.rho_air * self.A_frontal * self.C_d \
					* (long_vel ** 2) * self.R_tire 

		T_vector = np.array([[Tg], [Te], [Tm - (T_load + Tb) / self.final_gear], [0]])
		w_d = np.matmul(self.T2W, T_vector) # angular accelerations of plenetary gear

		return w_d # wg_d, we_d, wr_d, force_pinion

	def max_trq_E(self,we):
		if we < 1000*2*math.pi/60:
			maxtrq = 0
		# elif we > 420.0: # 418.9000
		# 	maxtrq = 0
		else:
			maxtrq = np.interp(we, self.eng_map_spd, self.eng_map_maxtrq)
		return maxtrq

	def engine(self, Te_des, we):
		# return engine torque (Nm) and fuel rate (g/s)
		if we < 1000*2*math.pi/60:
			maxtrq = 0
		else:
			maxtrq = np.interp(we, self.eng_map_spd, self.eng_map_maxtrq)
		Te = np.amax(np.amin([maxtrq, Te_des]), 0)
		FR = self.FR_calc(we, Te)[0]
		return Te, FR


	def max_trq_G(self,wg):
		maxtrq = np.interp(wg, self.g_max_spd, self.g_max_trq)
		return maxtrq

	def generator(self, Tg_des, wg):
		# return generator torque (Nm) and generator power (g/s)
		maxtrq = np.interp(wg, self.g_max_spd, self.g_max_trq)
		Tg = np.amin([np.abs(Tg_des), maxtrq]) * np.sign(Tg_des)

		Pg_idle = Tg * wg
		g_eff = self.g_eff_calc(wg, Tg)[0]

		if Pg_idle >= 0 :
			Pg = Pg_idle / g_eff # discharge (>0)
		else:
			Pg = Pg_idle * g_eff # charge (<0)

		return Tg, Pg # Generator torque (Nm), Generator Power (W)

	def max_trq_M(self, wr):
		maxtrq = np.interp(wr, self.m_map_spd, self.m_max_trq)
		return maxtrq

	def motor(self, Tm_des, wr):#ring gear speed = motor speed
		# return motor torque (Nm) and motor power (g/s)

		maxtrq = np.interp(wr, self.m_map_spd, self.m_max_trq)
		Tm = np.amin([np.abs(Tm_des), maxtrq]) * np.sign(Tm_des)

		Pm_idle = Tm * wr
		m_eff = self.m_eff_calc(wr, Tm)[0]

		if Pm_idle >= 0 :
			Pm = Pm_idle / m_eff # discharge (>0)
		else:
			Pm = Pm_idle * m_eff # charge (<0)

		return Tm, Pm # motor torque (Nm), motor Power (W)

	def battery(self, Pg, Pm, SOC):
		P_bat = Pg + Pm # Watt
		# Pack VOC, R (input : SOC, P_bat, mod temperature (C))
		#			  (output : VOC, R)
		VOC_1C = self.voc_calc(self.ess_fixtemp, SOC)[0] # VOC of 1 cell
		VOC = VOC_1C * self.ess_module_num

		if P_bat >= 0 : # discharge
			R = self.r_dis_calc(self.ess_fixtemp, SOC)[0]

		else:
			R = self.r_chg_calc(self.ess_fixtemp, SOC)[0]

		tmp = self.ess_res_scale_coef # just temperal copy for readability
		ess_res_scale = (tmp[0] * self.ess_module_num + tmp[1]) / (tmp[2] + self.ess_cap_scale + tmp[3])

		R = ess_res_scale * R

		# Limit Power (input : SOC, P_bat, VOC, R)
		#			  (output : P_bat_limited, ess_max_dis_pwr)
		eps = np.finfo(float).eps
		limit_ = np.amax([VOC * 0.5, self.m_min_volts, self.ess_module_num * self.ess_min_volts])
		# limit by battery max or min mc voltage or ess min
		ess_max_dis_pwr = (VOC - limit_) * limit_ / (R) -0.1
		# Discharge max power allowed

		if ((SOC >= 0.999) and (P_bat < 0)): #no power req if SOC>.999 and trying to charge
			P_bat_limited = 0
		else:
			if ((SOC > eps) or (P_bat < 0)): #no power req if SOC<=0 and trying to discharge
				P_bat_limited = np.amin([P_bat, ess_max_dis_pwr])
			else:
				P_bat_limited = 0


		# Compute Current (input : VOC, R, P_bat_limited)
		#				  (output : i_out, V_out, ess_max_chg_pwr)
		i_bat = (VOC - ((VOC ** 2) - (4 * R * P_bat_limited)) ** 0.5) / (2 * R)
		i_abs_min = (VOC - self.ess_max_volts * self.ess_module_num) / R
		if i_bat > i_abs_min:
			i_out = i_bat
		else:
			i_out = i_abs_min 

		v_out = VOC - i_out*R
		ess_max_chg_pwr = i_abs_min * self.ess_max_volts * self.ess_module_num

		# SOC Algorithm (input : i_out, ess_fixtemp)
		#				(output : SOC)
		max_cap = np.interp(self.ess_fixtemp, self.max_cap_map, self.ess_tmp)
		SOC_d = - i_out / 3600 / max_cap# Derivative of SOC
		# print(i_out)

		# print("Power:", Pg, Pm, P_bat, ess_max_dis_pwr, ess_max_chg_pwr)
		if (P_bat_limited <= ess_max_dis_pwr) and (P_bat_limited >= ess_max_chg_pwr):
			err_flag = False
		else:
			err_flag = True
		self.P_bat_limited = P_bat_limited
		return SOC_d, ess_max_dis_pwr, ess_max_chg_pwr, err_flag

	def calc_bound(self):

		G_trq_max = self.max_trq_G(self.wg)

		G_trq_min = - np.abs(G_trq_max)
		G_trq_max = np.abs(G_trq_max)
		E_trq_min = 0
		E_trq_max = self.max_trq_E(self.we)

		return G_trq_min, G_trq_max, E_trq_min, E_trq_max

	def SN(self, state): # state normalizer
		state = state.flatten()
		state[0] = state[0]/0.6265
		state[1] = state[1]/314.6437
		state[2] = (state[2]-261.8000)/92.9531
		state[3] = (state[3]-271.3194)/191.4084
		state[4] = (state[4]-0.55)/0.1291
		state[5] = (state[5]-8.7329)/6.5758
		state[6] = (state[6]-6.5758)/6.5758

		return state


	def calc_transition(self, acc, Tg, Te): #
		wr_d = acc * self.final_gear / self.R_tire

		[Tg, Pg] = self.generator(Tg, self.wg)
		[Te, FR] = self.engine(Te, self.we)

		long_vel = self.wr * self.R_tire/self.final_gear # (m/s) Longitudinal velocity
		f_rolling_flag = ((self.wr / self.final_gear) > 0.001)
		T_load = f_rolling_flag * self.f_rolling * self.mass * 9.81 * self.R_tire + 0.5 * self.rho_air * self.A_frontal * self.C_d \
					* (long_vel ** 2) * self.R_tire


		K_temp = (self.Ie + self. Ic) / ((self.Ig + self. Is) * (self.R + self.S))
		force_pinion = (Te - K_temp * self.S * Tg - K_temp * (self.Ig + self.Is) * self.R * wr_d) / (self.R + self.S + K_temp * (self.S ** 2))

		Tm = (self.I_veh + self.Im + self.Ir) * wr_d - self.R * force_pinion + (T_load) / self.final_gear
		[Tm, Pm] = self.motor(Tm, self.wr)

		[SOC_d, self.ess_max_dis_pwr, self.ess_max_chg_pwr, err_flag_P_bat] = self.battery(Pg, Pm, self.SOC) # Check memory!!!!!

		w_d = self.T2W_d(Tg, Te, Tm, 0, self.wr)
		if w_d[2,0] > wr_d:
			Tb = self.W2T[2,2] * (w_d[2,0] - wr_d) * self.final_gear
			w_d = self.T2W_d(Tg, Te, Tm, Tb, self.wr)

		if np.abs(w_d[2,0] - wr_d) > 0.001 :
			err_flag_wr_d = True
		else:
			err_flag_wr_d = False

		# if ((self.we + w_d[1,0] * self.dt) > 418.9000) and ((self.we + w_d[1,0] * self.dt) < 0):
		# 	err_we = True
		# else:
		# 	err_we = False

		# if np.abs((self.wg + w_d[0,0] * self.dt)) > 575.9587:
		# 	err_wg = True
		# else:
		# 	err_wg = False

		err_flag = (err_flag_wr_d or err_flag_P_bat) #  or err_we or err_wg

		action = np.array([Tg,Te])
		s_t_r = [w_d[0,0],w_d[1,0],w_d[2,0],w_d[3,0],FR,long_vel,SOC_d]
		#state transition rate 

		return s_t_r, action, err_flag

	##########################################
	def transition(self, s_t_r, action, end_flag): #

		wg_d = s_t_r[0]
		we_d = s_t_r[1]
		wr_d = s_t_r[2]
		Fp = s_t_r[3]
		FR = s_t_r[4]
		# long_vel = s_t_r[5]
		SOC_d =  s_t_r[6]

		cyc_d = np.interp(self.t, self.cyc_d[...,0].flatten(), self.cyc_d[...,1].flatten())

		mean_vel = np.mean(self.vel_queue)
		std_vel = np.std(self.vel_queue)

		state = np.array([cyc_d, self.wg, self.we, self.wr, self.SOC, mean_vel, std_vel])
		state_extra = np.array([self.fuel_consumption, self.distance])

		self.wg = self.wg + (wg_d) * self.dt

		self.wr = np.amax([self.wr + (wr_d) * self.dt, 0])
		self.we = np.amax([action[1], 0])
		

		if self.we > 418.9:
			self.we = 418.9
		elif self.we < 0:
			self.we = 0


		if self.wg > 575.9587:
			self.wg = 575.9587
		elif self.wg < -575.9587:
			self.wg = -575.9587		

		self.wr = ((self.R + self.S) * self.we - self.S * self.wg) / self.R
		# print('Kinematics',((self.R + self.S) * self.we - self.S * self.wg) - self.wr *self.R)
		long_vel = self.wr * self.R_tire/self.final_gear # (m/s) Longitudinal velocity

		# print(self.we, self.wg, self.wr)

		self.impulse_pinion = self.impulse_pinion + (Fp) * self.dt
		self.fuel_consumption = self.fuel_consumption + (FR) * self.dt
		self.distance = self.distance + (long_vel) * self.dt 
		self.SOC = self.SOC + (SOC_d) * self.dt


		cyc_d_next = np.interp(self.t + self.dt, self.cyc_d[...,0].flatten(), self.cyc_d[...,1].flatten())
		self.vel_queue.append(long_vel)
		while np.sum(self.vel_queue) * self.dt > 300:
			self.vel_queue.popleft()

		mean_vel = np.mean(self.vel_queue)
		std_vel = np.std(self.vel_queue)

		state_next = np.array([cyc_d_next, self.wg, self.we, self.wr, self.SOC, mean_vel, std_vel])
		state_extra_next = np.array([self.fuel_consumption, self.distance])

		if end_flag:
			reward = self.distance/self.fuel_consumption
			if self.SOC < 0.5:
				reward += - 5000*(0.55 - self.SOC)
			elif self.SOC > 0.6:
				reward += - 5000*(self.SOC - 0.55)
		else:
			reward = 0 
		if self.SOC < 0.4 or self.SOC > 0.7:
			reward =  - 1# SOC penalty

		return state, action, np.array(reward), state_next, state_extra, state_extra_next

		#return w_d[2,0]*self.R_tire/self.final_gear, Te, Tg, Tm, self.we, self.wg, self.wr, self.fuel_consumption, self.distance, self.SOC

	def feed_cyc(self):
		cyc_d = np.interp(self.t, self.cyc_d[...,0].flatten(), self.cyc_d[...,1].flatten())
		self.t = self.t + self.dt
		if self.t >= self.tend : 
			end_flag = True
		else:
			end_flag = False
		return cyc_d, end_flag

	def init(self, soc_init, cyc_num = 0):

		self.current_cyc = cyc_num
		self.tend = self.cyc_list[self.current_cyc][-1,0]
		self.t = 0

		#States

		self.wg = 0 # (rad/sec) Angular vel of generator
		self.we = 0*2*math.pi/60 # (rad/sec) Angular vel of engine
		self.wr = 0 # (rad/sec) Angular vel of ring gear of plenetery gear
		self.impulse_pinion = 0 # (N-sec)
		# impulse from pinion to ring gear or to sungear or to opposite and half to carrier

		self.fuel_consumption = 0
		self.distance = 0

		self.SOC = soc_init

		states = [0, self.wg, self.we, self.wr, self.SOC, 0, 0]
		states_extra = [self.fuel_consumption, self.distance]
		return states, self.tend


	def calc_bound2(self,acc):
		wr_d = acc * self.final_gear / self.R_tire

		Tg_max = self.max_trq_G(self.wg)
		Tg_min = - self.max_trq_G(self.wg)


		long_vel = self.wr * self.R_tire/self.final_gear # (m/s) Longitudinal velocity
		f_rolling_flag = ((self.wr / self.final_gear) > 0.001)
		T_load = f_rolling_flag * self.f_rolling * self.mass * 9.81 * self.R_tire + 0.5 * self.rho_air * self.A_frontal * self.C_d \
					* (long_vel ** 2) * self.R_tire

		Tr_max = self.max_trq_M(self.wr) - (T_load/self.final_gear)
		Tr_min = - self.max_trq_M(self.wr)- (T_load/self.final_gear)

		eng_cap_max1 = ((self.T2W[1,2] / self.T2W[2,2]) * wr_d) + (self.T2W[1,0] - ((self.T2W[2,0] * self.T2W[1,2]) / self.T2W[2,2])) * Tg_max + self.we
		eng_cap_min1 = ((self.T2W[1,2] / self.T2W[2,2]) * wr_d) + (self.T2W[1,0] - ((self.T2W[2,0] * self.T2W[1,2]) / self.T2W[2,2])) * Tg_min + self.we
		k1 = (self.T2W[2,1] * self.T2W[1,2] / self.T2W[2,2]) - self.T2W[1,1]

		eng_cap_max2 = ((self.T2W[1,0] / self.T2W[2,0]) * wr_d) + (self.T2W[1,2] - ((self.T2W[1,0] * self.T2W[2,2]) / self.T2W[2,0])) * Tr_max + self.we
		eng_cap_min2 = ((self.T2W[1,0] / self.T2W[2,0]) * wr_d) + (self.T2W[1,2] - ((self.T2W[1,0] * self.T2W[2,2]) / self.T2W[2,0])) * Tr_min + self.we
		k2 = (self.T2W[1,0] * self.T2W[2,1] / self.T2W[2,0]) - self.T2W[1,1]

		# print(eng_cap_max1, eng_cap_min1,eng_cap_max2, eng_cap_min2, k1, k2, Tr_max, Tr_min)
		# print('MIN',(eng_cap_max1 - eng_cap_max2)/(k1 - k2))
		# print('MAX',(eng_cap_min1 - eng_cap_min2)/(k1 - k2))
		# Te = 0
		# print(eng_cap_max1 - k1 * Te)
		# print(eng_cap_min1 - k1 * Te)
		# print(eng_cap_max2 - k2 * Te)
		# print(eng_cap_min2 - k2 * Te)



	def calc_bound_Te(self):

		Te_max = self.max_trq_E(self.we)
		Te_min = 0

		return Te_min, Te_max

	def calc_bound_we(self, acc, Te):
		wr_d = acc * self.final_gear / self.R_tire

		Tg_max1 = self.max_trq_G(self.wg)
		Tg_min1 = - self.max_trq_G(self.wg)

		Tm_max = self.max_trq_M(self.wr)
		Tm_min = - self.max_trq_M(self.wr)


		long_vel = self.wr * self.R_tire/self.final_gear # (m/s) Longitudinal velocity
		f_rolling_flag = ((self.wr / self.final_gear) > 0.001)
		T_load = f_rolling_flag * self.f_rolling * self.mass * 9.81 * self.R_tire + 0.5 * self.rho_air * self.A_frontal * self.C_d \
					* (long_vel ** 2) * self.R_tire

		Tg_max2 = (wr_d - (self.T2W[2,1] * Te) - (self.T2W[2,2] * (Tm_max - (T_load/self.final_gear)))) / self.T2W[2,0]
		Tg_min2 = (wr_d - (self.T2W[2,1] * Te) - (self.T2W[2,2] * (Tm_min - (T_load/self.final_gear)))) / self.T2W[2,0]

		Tg_max = np.amin([Tg_max1,Tg_max2])
		Tg_min = np.amax([Tg_min1,Tg_min2])

		# print('Tg:' ,Tg_max1, Tg_max2, Tg_min1, Tg_min2, Tg_max, Tg_min)

		# print((self.T2W[1,2] / self.T2W[2,2]), (self.T2W[1,1] - ((self.T2W[2,1] * self.T2W[1,2]) / self.T2W[2,2])), (self.T2W[1,0] - ((self.T2W[2,0] * self.T2W[1,2]) / self.T2W[2,2])))

		# print('Tg_gain', (self.T2W[1,0] - ((self.T2W[2,0] * self.T2W[1,2]) / self.T2W[2,2])))

		we_d_max = (self.T2W[1,2] / self.T2W[2,2]) * wr_d + (self.T2W[1,1] - ((self.T2W[2,1] * self.T2W[1,2]) / self.T2W[2,2])) * Te + \
				(self.T2W[1,0] - ((self.T2W[2,0] * self.T2W[1,2]) / self.T2W[2,2])) * Tg_max
		we_d_min = (self.T2W[1,2] / self.T2W[2,2]) * wr_d + (self.T2W[1,1] - ((self.T2W[2,1] * self.T2W[1,2]) / self.T2W[2,2])) * Te + \
				(self.T2W[1,0] - ((self.T2W[2,0] * self.T2W[1,2]) / self.T2W[2,2])) * Tg_min

		we_max1 = (self.we + we_d_max * self.dt)
		we_max2 = 418.9000
		we_max3 = ((self.wr + wr_d * self.dt) * self.R + 575.9587 * self.S) / (self.R + self.S)

		we_min1 = (self.we + we_d_min * self.dt)
		we_min2 = 0
		we_min3 = ((self.wr + wr_d * self.dt) * self.R - 575.9587 * self.S) / (self.R + self.S)

		we_max = np.amin([we_max1, we_max2, we_max3])
		we_min = np.amax([we_min1, we_min2, we_min3])

		# print('Te',Te)
		# print('we',self.we)
		# print('we list',[we_max1, we_max2, we_max3], [we_min1, we_min2, we_min3])
		# print('we result',we_max, we_min)
		# if np.abs(we_max) > 500:
		# 	print([we_max1, we_max2, we_max3], [we_min1, we_min2, we_min3])
		# 	print(we_max, we_min, Te)
		# 	print(Tg_max1, Tg_max2, Tg_min1, Tg_min2, Tm_max, Tm_min)
		# 	print(self.T2W[2,2], self.T2W[2,0])
		return we_min, we_max


	def calc_transition2(self, acc, Te, we_des): #
		wr_d = acc * self.final_gear / self.R_tire

		long_vel = self.wr * self.R_tire/self.final_gear # (m/s) Longitudinal velocity
		f_rolling_flag = ((self.wr / self.final_gear) > 0.001)
		T_load = f_rolling_flag * self.f_rolling * self.mass * 9.81 * self.R_tire + 0.5 * self.rho_air * self.A_frontal * self.C_d \
					* (long_vel ** 2) * self.R_tire

		we_d = (we_des - self.we) / self.dt
		wg_d = (((self.R + self.S) * we_d) - self.R * wr_d) / self.S
		Fp = (Te - ((self.Ie + self.Ic) * we_d)) / (self.R + self.S)

		T_vec = np.matmul(self.W2T, np.array([[wg_d, we_d, wr_d, Fp]]).transpose())

		Tg = T_vec[0,0]
		Te = T_vec[1,0]
		Tm = T_vec[2,0] + T_load/self.final_gear


		[Tg, Pg] = self.generator(Tg, self.wg)
		[Te, FR] = self.engine(Te, self.we)
		[Tm, Pm] = self.motor(Tm, self.wr)
		[SOC_d, self.ess_max_dis_pwr, self.ess_max_chg_pwr, err_flag_P_bat] = self.battery(Pg, Pm, self.SOC) # Check memory!!!!!
		maxtrq = np.interp(self.wg, self.g_max_spd, self.g_max_trq)
		# print(self.g_max_spd)
		# print(self.g_max_trq)
		# print('gen',Tg, maxtrq, self.wg)

		w_d = self.T2W_d(Tg, Te, Tm, 0, self.wr)
		if w_d[2,0] > wr_d:
			Tb = self.W2T[2,2] * (w_d[2,0] - wr_d) * self.final_gear
			w_d = self.T2W_d(Tg, Te, Tm, Tb, self.wr)

		if np.abs(w_d[2,0] - wr_d) > 0.001 :
			err_flag_wr_d = True
		else:
			err_flag_wr_d = False


		err_flag = (err_flag_wr_d or err_flag_P_bat) #  or err_we or err_wg

		action = np.array([Te,we_des])
		s_t_r = [w_d[0,0],w_d[1,0],w_d[2,0],w_d[3,0],FR,long_vel,SOC_d]
		return s_t_r, action, err_flag
