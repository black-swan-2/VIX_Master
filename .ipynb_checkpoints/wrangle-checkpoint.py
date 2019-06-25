def get_data():
    """
    Downloads and wrangles data for BlackSwan 2.0 VIX modeling. 
  
    Combines the two index datasets and creates a 
    new DataFrame (df) that contains the target feature (Trading Days with 3 
    STDV shifts based on a rolling window)
  
    :input: vix  => CBOE VIX Historical Data
    :input: gspc => GSPC S&P 500 index with matching start date to VIX
  
    :return: df  => cominbation VIX/GSPC Dataset with Target Classifier 
    """
  
    ############### Getting Data ##################
  
    vix = (data.DataReader('^VIX', 
                           "yahoo", 
                           start='1990-1-02', 
                           end='2019-6-17')
           .drop(columns = ['Volume', 'Adj Close']))
  
    gspc = data.DataReader('^GSPC', 
                           "yahoo", 
                           start='1990-1-02',
                           end='2019-6-17')
    
    treasury = (pd.read_csv('USTREASURY-YIELD.csv')
                .sort_values(by = 'Date')
                .drop(columns=['1 MO', '2 MO', '20 YR']))
  
    ############### Wrangling Data #################
    
    # Rename the Columns
    vix.columns      = ['vix_high', 'vix_low', 'vix_open', 'vix_close']
    gspc.columns     = ['gspc_high', 'gspc_low', 'gspc_open',
                        'gspc_close','gspc_volume','gspc_adj_close']
  
    # Join the VIX and GSPC
    df = vix.join(gspc)
  
    # Pull Date columns out of the index
    df = df.reset_index()
    
    # Merge DF with the Treasury Data on the Date Feature
    # Date needs to be converted to Datetime format to match df['Date']
    treasury['Date'] = pd.to_datetime(treasury['Date'],
                                      infer_datetime_format=True)
    
    df = pd.merge(df, treasury, how='inner', on='Date')
    
    # Datetime Conversion
    # df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format= True)
  
    ############### Momemntum Feature Engineering ################
  
    # Awesome Oscillator
    df['mom_ao']=ta.momentum.ao(df['gspc_high'],
                                df['gspc_low'],
                                s=5,len=34,
                                fillna=True)

    # Money Flow Index
    df['mom_mf']=ta.momentum.money_flow_index(df['gspc_high'],
                                              df['gspc_low'],
                                              df['gspc_close'],
                                              df['gspc_volume'],
                                              n=14,fillna=True)
  
    # Relative Strength Index
    df['mom_rsi'] = ta.momentum.rsi(df['gspc_close'],
                                    n=14,
                                    fillna=True)
  
    # Stochasitc Oscillator
    df['mom_stoch']=ta.momentum.stoch(df['gspc_high'],
                                      df['gspc_low'],
                                      df['gspc_close'],
                                      n=14,
                                      fillna=True)
  
    # Stochasitc Signal
    df['mom_st_sig']=ta.momentum.stoch_signal(df['gspc_high'],
                                              df['gspc_low'],
                                              df['gspc_close'],
                                              n=14,
                                              d_n=3,
                                              fillna=True)
  
    # True Strength Indicator
    df['mom_tsi'] = ta.momentum.tsi(df['gspc_close'],
                                    r=25,
                                    s=13,
                                    fillna=True)
  
    # Ultimate Oscillator
    df['mom_uo'] = ta.momentum.uo(df['gspc_high'],
                                  df['gspc_low'],
                                  df['gspc_close'], 
                                  s=7, 
                                  m=14, 
                                  len=28, 
                                  ws=4.0, 
                                  wm=2.0, 
                                  wl=1.0,
                                  fillna=True)
  
    # Williams %R
    df['mom_wr']=ta.momentum.wr(df['gspc_high'],
                                df['gspc_low'],
                                df['gspc_close'],
                                lbp=14,fillna=True)
  
    ############### Volume Feature Engineering ####################
  
    # Accumulation/Distribution Index
    df['vol_adi']=ta.volume.acc_dist_index(df['gspc_high'],
                                           df['gspc_low'],
                                           df['gspc_close'],
                                           df['gspc_volume'],
                                           fillna=True)
  
    # Chaikin Money Flow
    df['vol_cmf'] = ta.volume.chaikin_money_flow(df['gspc_high'],
                                                 df['gspc_low'],
                                                 df['gspc_close'],
                                                 df['gspc_volume'],
                                                 n=20,fillna=True)
  
    # Ease of Movement
    df['vol_eom'] = ta.volume.ease_of_movement(df['gspc_high'],
                                               df['gspc_low'],
                                               df['gspc_close'],
                                               df['gspc_volume'],
                                               n=20,fillna=True)
  
    # Force Index
    df['vol_fm'] = ta.volume.force_index(df['gspc_close'],
                                         df['gspc_volume'],
                                         n=2,fillna=True)
  
    # Negative Volume Index
    df['vol_nvi'] = ta.volume.negative_volume_index(df['gspc_close'],
                                                    df['gspc_volume'],
                                                    fillna=True)
  
    # On-Balance Volume
    df['vol_obv'] = ta.volume.on_balance_volume(df['gspc_close'],
                                                df['gspc_volume'],
                                                fillna=True)
  
    # Volume-Price Trend
    df['vol_vpt'] = ta.volume.volume_price_trend(df['gspc_close'],
                                                 df['gspc_volume'],
                                                 fillna=True)
    
    ############### Volatility Feature Engineering
    
    #Average True Range
    df['atr_low'] = ta.volatility.average_true_range(df['gspc_high'],
                                                     df['gspc_low'],
                                                     df['gspc_close'],
                                                     n=23)
    
    df['atr_high'] = ta.volatility.average_true_range(df['gspc_high'],
                                                      df['gspc_low'],
                                                      df['gspc_close'],
                                                      n=37)
  
    ############### Target Creation #################
  
    # Determine daily market movement between Close and Close
    df['vix_move']  = (1 - df['vix_close']
                       .shift(1)/df['vix_close'])
  
    df['gspc_move'] = (1 - df['gspc_close']
                       .shift(1)/df['gspc_close'])
  
  
  
    ##### 30 Day Rolling Average
  
    # Find the Standard Deviation based on a rolling average (year)
    df['vix_rolling_30'] = (df['vix_move']
                            .rolling(30).std())
  
    df['gspc_rolling_30'] = (df['gspc_move']
                             .rolling(30).std())
  
    # Create new target features based on Different STDEV thresholds
    df['3sd_move_30'] = np.where(abs(df['gspc_move'])
                                 >=3*df['gspc_rolling_30'],1,0)
  
    df['2.5sd_move_30'] = np.where(abs(df['gspc_move'])
                                   >=2.5*df['gspc_rolling_30'],1,0)
  
    df['2sd_move_30'] = np.where(abs(df['gspc_move'])
                                 >=2*df['gspc_rolling_30'],1,0)
  
    df['1.5sd_move_30'] = np.where(abs(df['gspc_move'])
                                   >=1.5*df['gspc_rolling_30'],1,0)
  
  
    ##### 90 Day Rolling Average
  
    # Find the Standard Deviation based on a rolling average (year)
    df['vix_rolling_90'] = (df['vix_move']
                            .rolling(90).std())
  
    df['gspc_rolling_90'] = (df['gspc_move']
                             .rolling(90).std())
  
    # Create new target features based on Different STDEV thresholds
    df['3sd_move_90'] = np.where(abs(df['gspc_move'])
                                 >=3*df['gspc_rolling_90'],1,0)
  
    df['2.5sd_move_90'] = np.where(abs(df['gspc_move'])
                                   >=2.5*df['gspc_rolling_90'],1,0)
  
    df['2sd_move_90'] = np.where(abs(df['gspc_move'])
                                 >=2*df['gspc_rolling_90'],1,0)
  
    df['1.5sd_move_90'] = np.where(abs(df['gspc_move'])
                                   >=1.5*df['gspc_rolling_90'],1,0)
  
  
    ##### 120 Day Rolling Average
  
    # Find the Standard Deviation based on a rolling average (year)
    df['vix_rolling_120'] = (df['vix_move']
                            .rolling(120).std())
  
    df['gspc_rolling_120'] = (df['gspc_move']
                             .rolling(120).std())
  
    # Create new target features based on Different STDEV thresholds
    df['3sd_move_120'] = np.where(abs(df['gspc_move'])
                                  >=3*df['gspc_rolling_120'],1,0)
  
    df['2.5sd_move_120'] = np.where(abs(df['gspc_move'])
                                    >=2.5*df['gspc_rolling_120'],1,0)
  
    df['2sd_move_120'] = np.where(abs(df['gspc_move'])
                                  >=2*df['gspc_rolling_120'],1,0)
  
    df['1.5sd_move_120'] = np.where(abs(df['gspc_move'])
                                    >=1.5*df['gspc_rolling_120'],1,0)
  
  
    ##### 252 Day Rolling Average
  
    # Find the Standard Deviation based on a rolling average (year)
    df['vix_rolling_252'] = (df['vix_move']
                            .rolling(252).std())
  
    df['gspc_rolling_252'] = (df['gspc_move']
                             .rolling(252).std())
  
    # Create new target features based on Different STDEV thresholds
    df['3sd_move_252'] = np.where(abs(df['gspc_move'])
                                  >=3*df['gspc_rolling_252'],1,0)
  
    df['2.5sd_move_252'] = np.where(abs(df['gspc_move'])
                                    >=2.5*df['gspc_rolling_252'],1,0)
  
    df['2sd_move_252'] = np.where(abs(df['gspc_move'])
                                  >=2*df['gspc_rolling_252'],1,0)
  
    df['1.5sd_move_252'] = np.where(abs(df['gspc_move'])
                                    >=1.5*df['gspc_rolling_252'],1,0)
  
  
    ############## Handling Null Values ##################
    
    # Interpolating the Null Values for 30yr Treasury Bonds
    df['30 YR'] = (df['30 YR'].interpolate(method='spline',
                                           order=4))
    
    # Drop the rest
    df = df.dropna()
  
    return df
  