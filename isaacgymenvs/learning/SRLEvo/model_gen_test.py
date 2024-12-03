'''
MJCF模型生成测试
'''

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model_grammar')))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from model_grammar import SRL_mode1, SRL_mode2, ModelGenerator

    srl_mode = 'mode1'
    # name = 'humanoid_srl_mode1_p1'
    name = 'hsrl_GA_314_iter37'
    pretrain = False
    # base_srl_params = {
    #                 "first_leg_lenth" : 0.40,
    #                 "first_leg_size"  : 0.03,
    #                 "second_leg_lenth": 0.80,
    #                 "second_leg_size" : 0.03,
    #                 "third_leg_size"  : 0.05,
    #             }    
    # pretrian_srl_params = {
    #                 "first_leg_lenth" : 0.01,
    #                 "first_leg_size"  : 0.01,
    #                 "second_leg_lenth": 0.01,
    #                 "second_leg_size" : 0.01,
    #                 "third_leg_size"  : 0.01,
    #             }    
    srl_params = {
                    "first_leg_lenth" : 0.38374,
                    "first_leg_size"  : 0.0270,
                    "second_leg_lenth": 0.8942,
                    "second_leg_size" : 0.039,
                    "third_leg_size"  : 0.0231,
                }    
    srl_generator = { "mode1": SRL_mode1 ,
                      "mode2": SRL_mode2 ,}[srl_mode]
    srl_R = srl_generator( name=name, pretrain=pretrain, **srl_params)

    # 使用绝对路径来确定 save_path
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_path, '../../../assets/mjcf/humanoid_srl/')

    mjcf_generator = ModelGenerator(srl_R,save_path=save_path)
    back_load = not pretrain
    mjcf_generator.gen_basic_humanoid_xml()
    mjcf_generator.get_SRL_dfs(back_load=back_load)
    mjcf_generator.generate()



    srl_mode = 'mode2'
    name = 'humanoid_srl_mode2'
    pretrain = False
 
    srl_generator = { "mode1": SRL_mode1 ,
                      "mode2": SRL_mode2 ,}[srl_mode]
    srl_R = srl_generator( name=name, pretrain=pretrain)

    # 使用绝对路径来确定 save_path
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_path, '../../../assets/mjcf/humanoid_srl/')

    mjcf_generator = ModelGenerator(srl_R,save_path=save_path)
    back_load = not pretrain
    mjcf_generator.gen_basic_humanoid_xml()
    mjcf_generator.get_SRL_dfs(back_load=back_load)
    mjcf_generator.generate()