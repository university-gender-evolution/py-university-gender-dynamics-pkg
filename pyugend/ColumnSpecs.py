

MODEL_RUN_COLUMNS = list(['number_f1',
                          'number_f2',
                          'number_f3',
                          'number_m1',
                          'number_m2',
                          'number_m3',
                          'attrition_3',
                          'attrition_2',
                          'attrition_1',
                          'prom1',
                          'prom2',
                          'gender_proportion_overall',
                          'unfilled_vacancies',
                          'department_size',
                          'f_hire_3',
                          'm_hire_3',
                          'f_hire_2',
                          'm_hire_2',
                          'f_hire_1',
                          'm_hire_1',
                          'f_prom_3',
                          'm_prom_3',
                          'f_prom_2',
                          'm_prom_2',
                          'f_prom_1',
                          'm_prom_1'])


EXPORT_COLUMNS_FOR_CSV = list(['hiring_rate_women_1',
                               'hiring_rate_women_2',
                               'hiring_rate_women_3',
                               'hiring_rate_men_1',
                               'hiring_rate_men_2',
                               'hiring_rate_men_3',
                               'attrition_rate_women_1',
                               'attrition_rate_women_2',
                               'attrition_rate_women_3',
                               'attrition_rate_men_1',
                               'attrition_rate_men_2',
                               'attrition_rate_men_3',
                               'probablity_of_outside_hire_1',
                               'probability_of_outside_hire_2',
                               'probability_of_outside_hire_3',
                               'female_promotion_rate_1',
                               'female_promotion_rate_2',
                               'male_promotion_rate_1',
                               'male_promotion_rate_2',
                               'dept_size_upperbound',
                               'dept_size_lowerbound',
                               'dept_size_exogenous_variation_range',
                               'duration'])


RESULTS_COLUMNS = list(['year', 'mean_f1', 'mean_f2',
                        'mean_f3', 'mean_m1',
                        'mean_m2', 'mean_m3',
                        'mean_vac_3', 'mean_vac_2',
                        'mean_vac_1', 'mean_prom1',
                        'mean_prom2', 'mean_gendprop',
                        'mean_unfilled', 'mean_dept_size',
                        'mean_f_hire_3', 'mean_m_hire_3',
                        'mean_f_hire_2', 'mean_m_hire_2',
                        'mean_f_hire_1', 'mean_m_hire_1',
                        'mean_f_prom_3', 'mean_m_prom_3',
                        'mean_f_prom_2', 'mean_m_prom_2',
                        'mean_f_prom_1', 'mean_m_prom_1',
                        'std_f1', 'std_f2',
                        'std_f3', 'std_m1',
                        'std_m2', 'std_m3',
                        'std_vac_3', 'std_vac_2',
                        'std_vac_1', 'std_prom1',
                        'std_prom2', 'std_gendprop',
                        'std_unfilled', 'std_dept_size',
                        'std_f_hire_3', 'std_m_hire_3',
                        'std_f_hire_2', 'std_m_hire_2',
                        'std_f_hire_1', 'std_m_hire_1',
                        'std_f_prom_3', 'std_m_prom_3',
                        'std_f_prom_2', 'std_m_prom_2',
                        'std_f_prom_1', 'std_m_prom_1',
                        'f1_025', 'f2_025',
                        'f3_025', 'm1_025',
                        'm2_025', 'm3_025', 'vac_3_025',
                        'vac_2_025', 'vac_1_025',
                        'prom1_025', 'prom2_025',
                        'gendprop_025', 'unfilled_025',
                        'dept_size_025', 'f_hire_3_025',
                        'm_hire_3_025', 'f_hire_2_025',
                        'm_hire_2_025', 'f_hire_1_025',
                        'm_hire_1_025', 'f_prom_3_025',
                        'm_prom_3_025', 'f_prom_2_025',
                        'm_prom_2_025', 'f_prom_1_025',
                        'm_prom_1_025', 'f1_975',
                        'f2_975', 'f3_975',
                        'm1_975', 'm2_975',
                        'm3_975', 'vac_3_975',
                        'vac_2_975', 'vac_1_975',
                        'prom_1_975', 'prom2_975',
                        'gendprop_975', 'unfilled_975',
                        'dept_size_975', 'f_hire_3_975',
                        'm_hire_3_975', 'f_hire_2_975',
                        'm_hire_2_975', 'f_hire_1_975',
                        'm_hire_1_975', 'f_prom_3_975',
                        'm_prom_3_975', 'f_prom_2_975',
                        'm_prom_2_975', 'f_prom_1_975',
                        'm_prom_1_975'
                        ])

FEMALE_MATRIX_COLUMNS = list(['year',
                              'mpct_f1',
                              'spct_f1',
                              'mpct_f2',
                              'spct_f2',
                              'mpct_f3',
                              'spct_f3',
                              'mpct_m1',
                              'spct_m1',
                              'mpct_m2',
                              'spct_m2',
                              'mpct_m3',
                              'spct_m3',
                              'pf1_025',
                              'pf1_975',
                              'pf2_025',
                              'pf2_975',
                              'pf3_025',
                              'pf3_975',
                              'pm1_025',
                              'pm1_975',
                              'pm2_025',
                              'pm2_975',
                              'pm3_025',
                              'pm3_975'])
