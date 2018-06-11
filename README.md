# RobotRecognitionSystem-Termproject

1. 프로젝트 개요
 - 프로젝트 명 : DNN 기반 하이브리드 동력 분배 알고리즘 개발
 - 연구 목적 : Dynamic programming으로 얻은 데이터를 통해 연비를 좋게하는 하이브리드 동력 분배 알고리즘 개발

2. 사용한 데이터 셋
 - Dynamic programming을 통해 얻은 데이터셋 : 각 폴더에서 데이터 설명
 - 라벨 정보 : engine torque/generator torque(회귀 문제), engine on/off 분류라벨(분류 문제)

3. 사용한 모델 : Supervised learning(Fully connected layer/Convolution neural network/Long short term memory/Gated recurrent unit)
                Reinforcement learning(Deep deterministic policy gradient - Fully connected layer)
             
4. 딥 러닝 라이브러리 : Tensorflow

5. 실험 및 결과 요약
 - Supervised learning: Engine Torque와 Generator Torque Regression 문제로 FCL/CNN/LSTM/GRU 다 활용한 결과, FCL을 사용하는 것이 가장 결과가 좋게나왔다.
   추가적으로 회귀 문제 전에 Engine Torque를 on/off로 분류하여 더 높은 결과를 획득하고자 하였다. 이 역시 FCL일 때 가장 좋은 결과를 얻을 수 있었다.
   따라서, 최종 동력 분배 문제는 FCL 기반으로 Engine의 on/off를 분류하고, Engine off 일 때는 Engine torque=0으로, Generator torque를 FCL로 예측하고, 
   Engine on일 때는 Engine torque와 Generator torque 둘 다 FCL로 예측하였고 이를 통해 실시간으로 돌린 결과 DP 대비 90% 이상의 좋은 연비를 획득할 수 있었다.
 - Reinforcement learning: 논문을 참고하여 Deep Deterministic Policiy Gradient 모듈을 구현하였고 OpenAI gym 라이브러리의 pendulum enviroment로 검증 결과 정상적으로 동작하였다.
   이에 하이브리드 차량 모델을 적용하여 최적의 Action(Engine Torque, Engin Speed)를 학습하도록 알고리즘을 돌린 결과, 네트워크가 수렴하지 않았고 가장 연비가 좋은 경우도 supervised learning의 절반 정도의 성능을 보였다.
   이는 크게 두가지 원인이 있는것으로 보인다. 
   1) 불필요한 데이터 학습: exploration으로 인해 발생한 작동 불가능한 경우 (SOC 제한 위반, 속도 제한 위반 등)를 함께 사용함으로써 무의미한 학습이 진행됨
   2) 상태를 대표하지 못하는 state: 설정한 state가 현재 상황을 충분히 반영하지 못해 학습 결과가 수렴하지 못한것으로 보임

6. 같은 폴더에 있으면 돌아가게끔 정리하였습니다. 
