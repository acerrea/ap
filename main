import requests
import pandas as pd
import jdatetime
from jdatetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import sys # برای خروج از برنامه در صورت نبود متغیرهای محیطی

# --- (بخش ۱) تنظیمات امن و آماده برای گیت‌هاب ---

# خواندن اطلاعات حساس از متغیرهای محیطی
# این کار از قرار گرفتن توکن و آیدی در کد جلوگیری می‌کند
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PROXY_URL = os.getenv('IRANIAN_PROXY_URL') # مثلا: 'http://user:pass@1.2.3.4:8080'

# بررسی وجود متغیرهای ضروری
if not BOT_TOKEN or not CHAT_ID:
    print("خطا: متغیرهای محیطی TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID تنظیم نشده‌اند.")
    print("لطفاً یک فایل .env بسازید و مقادیر را در آن قرار دهید.")
    sys.exit(1) # خروج از برنامه با کد خطا

# تنظیمات پروکسی برای درخواست‌های اینترنتی
proxies = None
if PROXY_URL:
    proxies = {
        'http': PROXY_URL,
        'https': PROXY_URL,
    }
    print(f"-> در حال استفاده از پروکسی: {PROXY_URL}")
else:
    print("-> بدون پروکسی.")


def send_to_telegram_api(image_path, caption_text):
    """این تابع عکس و متن را با استفاده از پروکسی (در صورت وجود) به تلگرام ارسال می‌کند."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            data = {'chat_id': CHAT_ID, 'caption': caption_text, 'parse_mode': 'HTML'}
            # ارسال درخواست با پروکسی
            response = requests.post(url, files=files, data=data, proxies=proxies, timeout=30)
            response_json = response.json()
            if response_json.get("ok"):
                print(" -> پیام با موفقیت به تلگرام ارسال شد.")
            else:
                print(f"ERROR sending message to Telegram: {response_json.get('description')}")
    except Exception as e:
        print(f"An exception occurred while sending to Telegram: {e}")

# --- (بخش ۲) تنظیمات اولیه و داده‌های ثابت (بدون تغییر) ---
no = datetime.now()
now = f'{no : %Y/%m/%d - %H:%M:%S }'
now1 = f'{no : %Y-%m-%d}'

historical_volatility_map = {'خبهمن': 0.3457, 'وبملت': 0.3859, 'وبصادر': 0.3591, 'وتجارت': 0.3504,
                             'فولاد': 0.3317, 'خگستر': 0.3774, 'خودرو': 0.5927, 'فملي': 0.2988,
                             'شپنا': 0.3623, 'خساپا': 0.6239, 'شستا': 0.3349, 'ذوب': 0.351,
                             'سامان': 0.3412, 'بساما': 0.2906, 'خاور': 0.3506, 'كرمان': 0.3608,
                             'كروميت': 0.387, 'فزر': 0.3121, 'فسوژ': 0.3367, 'وتعاون': 0.3391,
                             'خپارس': 0.3378, 'اهرم': 0.409, 'بيدار': 0.4245, 'جهش': 0.4284,
                             'خودران': 0.3536, 'شتاب': 0.4326, 'هموزن': 0.3018, 'موج': 0.425,
                             'نارنج': 0.4429, 'پادا': 0.2646, 'پناه': 0.2684, 'پتروپاداش': 0.2654,
                             'پتروآبان': 0.3036, 'رويين': 0.2815, 'ثمين': 0.2791, 'اطلس': 0.279,
                             'آساس': 0.2655, 'تيام': 0.2525, 'توان': 0.4158,'اخابر': 0.4098}


default_historical_sigma = 0.35
IV_LOOKBACK_DAYS = 5
IV_WARNING_THRESHOLD_PERCENT = 35.0
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0'}
url_tse_options = 'https://cdn.tsetmc.com/api/Instrument/GetInstrumentOptionMarketWatch/0'

# --- (بخش ۳) توابع بهینه‌سازی شده بلک-شولز و یونانی‌ها (بدون تغییر) ---
EPSILON = 1e-9

def calculate_greeks_and_price(S, K, T, r, sigma, option_type='call'):
    T = max(T, EPSILON); sigma = max(sigma, EPSILON); S = max(S, EPSILON); K = max(K, EPSILON)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta_annual = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    else: # 'put'
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta_annual = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_daily = theta_annual / 365
    
    return {'price': max(price, 0), 'delta': delta, 'gamma': gamma, 'theta': theta_daily, 'vega': vega}

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    if option_type == 'call':
        min_price = max(0, S * np.exp(0) - K * np.exp(-r * T)) # More accurate lower bound
        if option_price < min_price - EPSILON:
            return 0.0
            
    option_price = max(option_price, EPSILON)
    func_to_solve = lambda sigma_iv: calculate_greeks_and_price(S, K, T, r, sigma_iv, option_type)['price'] - option_price
    try:
        implied_vol_val = newton(func_to_solve, x0=0.5, tol=1e-5, maxiter=100)
        return implied_vol_val if implied_vol_val > 1e-6 else 0.0
    except (RuntimeError, OverflowError, ValueError):
        return 0.0

# --- (بخش ۴) کد اصلی ---
print("شروع پردازش اطلاعات از TSETMC...")
try:
    # دریافت اطلاعات بازار با استفاده از پروکسی (در صورت وجود)
    response = requests.get(url_tse_options, headers=header, proxies=proxies, timeout=20)
    response.raise_for_status() # اگر خطایی (مثل 403 یا 500) رخ دهد، برنامه متوقف می‌شود
    r = response.text.split('},{')
    print(f"تعداد {len(r)} اختیار معامله دریافت شد.")
except requests.exceptions.RequestException as e:
    print(f"خطا در دریافت اطلاعات از TSETMC: {e}")
    print("ممکن است نیاز به پروکسی ایرانی داشته باشید یا سایت در دسترس نباشد.")
    sys.exit(1)


main_folder = now1
os.makedirs(main_folder, exist_ok=True)

swing_opportunities_folder = os.path.join(main_folder, "Swing_Trading_Opportunities")
os.makedirs(swing_opportunities_folder, exist_ok=True)

for i in r:
    try:
        if '"insCode_C":"' not in i: continue
        
        # ... (بقیه کدهای استخراج داده بدون تغییر) ...
        code = i.split('"insCode_C":"')[1].split('"')[0]
        nemad = i.split('"lVal18AFC_C":"')[1].split('"')[0]
        sherkat = i.split('"lVal30_C":"')[1].split('"')[0].split('-')[0]
        geymat_payani = int(i.split('"pClosing_C":')[1].split(',')[0])
        gp_nemad_asli = int(i.split('"pClosing_UA":')[1].split(',')[0])
        if geymat_payani == 0 or gp_nemad_asli == 0: continue
        best_bid_price = int(i.split('"pMeDem_C":')[1].split(',')[0])
        best_ask_price = int(i.split('"pMeOf_C":')[1].split(',')[0])
        bid_ask_spread_percent = ((best_ask_price - best_bid_price) / best_ask_price) * 100 if best_ask_price > 0 else 0
        identified_base_symbol = None; longest_match_len = 0
        for key in historical_volatility_map.keys():
            if key in sherkat and (identified_base_symbol is None or len(key) > longest_match_len):
                identified_base_symbol = key; longest_match_len = len(key)
        selected_historical_sigma = historical_volatility_map.get(identified_base_symbol, default_historical_sigma)
        geymat_emal = int(i.split('"lVal30_C":"')[1].split('"')[0].split('-')[1])
        tarikh_emal = i.split('"lVal30_C":"')[1].split('"')[0].split('-')[-1]
        andaze_garardad = int(i.split('"contractSize":')[1].split(',')[0])
        arzesh_moamelat = int(i.split('"qTotCap_C":')[1].split('.')[0])
        akherin_geymat = int(i.split('"pDrCotVal_C":')[1].split(',')[0])
        mogeyyat_baz = int(i.split('"oP_C":')[1].split(',')[0])
        roozhaye_bagimande = int(i.split('"remainedDay":')[1].split(',')[0])
        print(f"\nدر حال پردازش اختیار خرید: {nemad}")

        # دریافت اطلاعات تاریخچه قیمت با پروکسی (در صورت وجود)
        url_history = f'https://members.tsetmc.com/tsev2/chart/data/Financial.aspx?i={code}&t=ph&a=1'
        history_response = requests.get(url=url_history, headers=header, proxies=proxies, timeout=15)
        history_response.raise_for_status()
        g = history_response.text.split(';')
        
        data = [{'Date': x.split(',')[0], 'Open': int(x.split(',')[3]), 'Close': int(x.split(',')[4]), 'High': int(x.split(',')[1]), 'Low': int(x.split(',')[2]), 'volume': int(x.split(',')[5]), "پایانی": int(x.split(',')[6]), 'ارزش معاملات': (int(x.split(',')[5]) * int(x.split(',')[6])) * andaze_garardad} for x in g if len(x.split(',')) > 6]
        if not data: continue
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df['Shamsi_Date'] = df['Date'].apply(lambda x: jdatetime.date.fromgregorian(date=x).strftime('%Y/%m/%d'))
        
        # ... (بقیه کد شما بدون هیچ تغییری ادامه می‌یابد) ...
        S = gp_nemad_asli; K = geymat_emal; T = roozhaye_bagimande / 365.0; r = 0.30
        option_type = 'call'; sigma_manual = selected_historical_sigma
        greeks_manual = calculate_greeks_and_price(S, K, T, r, sigma_manual, option_type)
        bs_price_manual = greeks_manual['price']; delta_manual = greeks_manual['delta']; gamma_manual = greeks_manual['gamma']
        theta_manual = greeks_manual['theta']; vega_manual = greeks_manual['vega']
        
        implied_vol = implied_volatility(geymat_payani, S, K, T, r, option_type)
        if implied_vol > 1e-6:
            greeks_implied = calculate_greeks_and_price(S, K, T, r, implied_vol, option_type)
            delta_implied, gamma_implied, theta_implied, vega_implied = greeks_implied['delta'], greeks_implied['gamma'], greeks_implied['theta'], greeks_implied['vega']
            leverage_implied = abs(delta_implied * (S / geymat_payani)) if geymat_payani > EPSILON else 0
        else:
            delta_implied, gamma_implied, theta_implied, vega_implied, leverage_implied = 0, 0, 0, 0, 0
        
        is_swing_opportunity = False
        moneyness_ratio = S / K
        
        if (option_type == 'call' and
            5 <= roozhaye_bagimande <= 90 and
            arzesh_moamelat > 100_000_000 and
            bid_ask_spread_percent < 5.0 and
            0.90 <= moneyness_ratio <= 1.15 and
            gamma_manual > 0.0004 and
            implied_vol > 0 and implied_vol < 0.85):
            is_swing_opportunity = True
            print(f"SUCCESS: Found potential SWING TRADING opportunity: '{nemad}'")

        if is_swing_opportunity:
            days_for_scenario = 3
            T_scenario = max(0, roozhaye_bagimande - days_for_scenario) / 365.0
            S_optimistic = S * 1.05
            price_optimistic_bs = calculate_greeks_and_price(S_optimistic, K, T_scenario, r, sigma_manual, option_type)['price']
            S_realistic = S * 1.03
            price_realistic_bs = calculate_greeks_and_price(S_realistic, K, T_scenario, r, sigma_manual, option_type)['price']
            S_conservative = S
            price_conservative_bs = calculate_greeks_and_price(S_conservative, K, T_scenario, r, sigma_manual, option_type)['price']
            price_optimistic_intrinsic = max(0, S_optimistic - K)
            change_in_S_optimistic = S_optimistic - S
            price_optimistic_delta_approx = geymat_payani + (delta_manual * change_in_S_optimistic)
            change_in_S_realistic = S_realistic - S
            price_realistic_delta_approx = geymat_payani + (delta_manual * change_in_S_realistic)
            vega_impact_5_percent = vega_manual * 5
            
            iv_trend_text = "داده تاریخی کافی نیست"
            iv_spike_warning_text = ""
            try:
                if len(df) >= IV_LOOKBACK_DAYS + 1:
                    iv_values = []
                    S_used = S
                    for days_back in range(1, IV_LOOKBACK_DAYS + 1):
                        row = df.iloc[-(days_back + 1)]
                        historical_option_price = row.get('پایانی', 0)
                        if historical_option_price is None or historical_option_price <= 0:
                            continue
                        historical_days_remaining = roozhaye_bagimande + days_back
                        historical_T = max(historical_days_remaining / 365.0, EPSILON)
                        hist_iv = implied_volatility(historical_option_price, S_used, K, historical_T, r, option_type)
                        if hist_iv > 1e-6:
                            iv_values.append(hist_iv)
                    if implied_vol > 1e-6 and len(iv_values) >= 2:
                        iv_mean_hist = np.mean(iv_values)
                        iv_change_percent = ((implied_vol - iv_mean_hist) / iv_mean_hist) * 100
                        if iv_mean_hist > 0:
                            iv_spike_percent = ((implied_vol - iv_mean_hist) / iv_mean_hist) * 100
                            if iv_spike_percent > IV_WARNING_THRESHOLD_PERCENT:
                                iv_spike_warning_text = (f"\n🚨 <b>هشدار گرانی نوسان (IV Spike):</b>\n"
                                                         f"نوسان ضمنی فعلی (<b>{implied_vol*100:.1f}%</b>) حدود <b>{iv_spike_percent:.0f}%</b> بالاتر از میانگین تاریخی اخیر است. "
                                                         f"این موضوع ریسک خرید را به دلیل احتمال افت قیمت اختیار (IV Crush) افزایش می‌دهد.\n")
                        x = np.arange(len(iv_values))
                        y = np.array(iv_values)
                        A = np.vstack([x, np.ones_like(x)]).T
                        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                        daily_pct_slope = (slope / iv_mean_hist) * 100 if iv_mean_hist > 0 else 0
                        iv_slope_text = f" / شیب: {daily_pct_slope:+.1f}%"
                        if daily_pct_slope > 0.5:
                            iv_trend_text = f"<b>روند صعودی ✅</b> (بالاتر از میانگین {iv_change_percent:+.1f}%){iv_slope_text}"
                        elif daily_pct_slope < -0.5:
                             iv_trend_text = f"<b>روند نزولی ❌</b> (بالاتر از میانگین {iv_change_percent:+.1f}%){iv_slope_text}"
                        else:
                            iv_trend_text = f"روند خنثی ↔️ (بالاتر از میانگین {iv_change_percent:+.1f}%){iv_slope_text}"
                    elif len(iv_values) > 0:
                         iv_trend_text = "داده تاریخی برای تحلیل روند کافی نیست"
                    else:
                        iv_trend_text = "قیمت‌های تاریخی نامعتبر"
            except Exception as iv_e:
                print(f"ERROR calculating IV trend for {nemad}: {iv_e}")
                iv_trend_text = "خطا در محاسبه"
            
            def format_number_short(value):
                if abs(value) >= 10000000000: return f"{value / 10000000000:.1f} ميليارد ت"
                if abs(value) >= 10000000: return f"{value / 10000000:.1f} ميليون ت"
                return f'{value:,}'
            formatted_arzesh_short = format_number_short(arzesh_moamelat)
            hashtags = "#اختیار_خرید #نوسان_گیری #تحلیل_آپشن"
            moneyness_status = "بی‌تفاوت (ATM)" if 0.98 <= moneyness_ratio <= 1.02 else "نزدیک به ATM"
            telegram_caption = (
                f"🎯 <b>فرصت نوسان‌گیری شناسایی شد</b>\n\n"
                f"<b>نماد:</b> #{nemad} ({sherkat})\n"
                f"<b>قیمت فعلی اختیار:</b> {geymat_payani:,.0f} ریال\n\n"
                f"<b>تحلیل کلیدی نوسان‌گیری:</b>\n"
                f"▫️ <b>وضعیت:</b> {moneyness_status}\n"
                f"▫️ <b>زمان:</b> {roozhaye_bagimande} روز تا سررسید\n"
                f"▫️ <b>شتاب (گاما):</b> <code>{gamma_manual:.4f}</code> (پتانسیل رشد تصاعدی)\n"
                f"▫️ <b>نوسان ضمنی (IV):</b> <code>{implied_vol*100:.1f}%</code>\n"
                f"▫️ <b>روند نوسان ضمنی ({IV_LOOKBACK_DAYS} روزه):</b> {iv_trend_text}\n"
                f"▫️ <b>نقدشوندگی:</b> اسپرد {bid_ask_spread_percent:.1f}% | ارزش معاملات: <b>{formatted_arzesh_short}</b>\n"
                f"{iv_spike_warning_text}"
                f"\n📈 <b>سناریوهای قیمت اختیار (در ۳ روز آینده):</b>\n"
                f"<i>(محاسبات دقیق بر اساس مدل بلک-شولز انجام شده)</i>\n"
                f"🟢 <b>خوشبینانه (+۵٪ رشد سهم):</b> ~<b>{price_optimistic_bs:,.0f} ریال</b>\n"
                f"   (<i>مقایسه: تقریب دلتا: {price_optimistic_delta_approx:,.0f} | ارزش ذاتی: {price_optimistic_intrinsic:,.0f}</i>)\n"
                f"🟡 <b>واقع‌بینانه (+۳٪ رشد سهم):</b> ~<b>{price_realistic_bs:,.0f} ریال</b>\n"
                f"   (<i>مقایسه: تقریب دلتا: {price_realistic_delta_approx:,.0f}</i>)\n"
                f"🔴 <b>محافظه‌کارانه (عدم تغییر):</b> ~<b>{price_conservative_bs:,.0f} ریال</b> (اثر تتا)\n\n"
                f"⏳ <b>ریسک فرسایش زمانی (تتا):</b>\n"
                f"هر روز حدود <b>{abs(theta_manual):.1f} ریال</b> از ارزش اختیار کاسته می‌شود (با فرض ثبات سایر عوامل).\n\n"
                f"🌊 <b>حساسیت به نوسان (وگا):</b>\n"
                f"هر ۵٪ تغییر در نوسان ضمنی (IV) بازار، می‌تواند قیمت اختیار را حدود <b>±{vega_impact_5_percent:,.0f} ریال</b> جابجا کند.\n\n"
                f"⚠️ <b>هشدار:</b> این تحلیل بر اساس مدل بلک-شولز بوده و ریسک‌های بازار در آن لحاظ نشده است.\n\n"
                f"{hashtags}"
            )
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.9, 0.3])
            fig.add_trace(go.Candlestick(x=df['Shamsi_Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='قیمت آپشن'), row=1, col=1)
            fig.add_trace(go.Bar(x=df['Shamsi_Date'], y=df['volume'], name='حجم', marker_color='rgba(0, 128, 0, 0.5)'), row=2, col=1)
            last_date = df['Shamsi_Date'].iloc[-1]; last_hajm = df['volume'].iloc[-1]
            last_high = df['High'].iloc[-1]; last_low = df['Low'].iloc[-1]
            sarbesar = geymat_emal + geymat_payani
            navasan = (last_high - last_low) / last_low * 100 if last_low > 0 else 0
            percentage_diff = ((geymat_payani - bs_price_manual) / bs_price_manual) * 100 if bs_price_manual > EPSILON else float('inf')
            percentage_akharin = ((akherin_geymat - bs_price_manual) / bs_price_manual) * 100 if bs_price_manual > EPSILON else float('inf')
            leverage_manual = abs(delta_manual * (S / geymat_payani)) if geymat_payani > EPSILON else 0
            clean_annotations = [
                dict(text=f"شرکت : {sherkat} - {nemad} | تاريخ اعمال : {tarikh_emal} ، تعداد روز باقیمانده : {roozhaye_bagimande} روز | قيمت اعمال : {geymat_emal:,} ریال | قیمت نماد پایه : {gp_nemad_asli:,} ریال | موقعیت باز : {mogeyyat_baz:,} | دیتای دریافتی : {df.shape[0]} روز | بروزرسانی : {now}", xref="paper", yref="paper", x=0.5, y=1.06, showarrow=False, font=dict(family="Vazirmatn FD ExtraBold, sans-serif", size=16, color="#8400ff")),
                dict(text=f"تابلوی معاملات پريميوم مربوط به : {last_date} | ق پایانی : {geymat_payani:,} ریال | ق سربسر : {sarbesar:,} ریال | آخرین ق : {akherin_geymat:,} ریال | کمترین ق : {last_low:,} ریال | بیشترین ق : {last_high:,} ریال | درصد نوسان : {round(navasan, 2)}% | حجم معاملات : {last_hajm:,} | ارزش معاملات : {formatted_arzesh_short}", xref="paper", yref="paper", x=0.5, y=1.03, showarrow=False, font=dict(family="Vazirmatn FD ExtraBold, sans-serif", size=16, color="blue")),
                dict(text=f"بلک شولز ( قیمت منصفانه با سیگمای تاریخی {selected_historical_sigma * 100:.1f} %) : {bs_price_manual:,.0f} ریال | درصد اختلاف پرمیوم با این قیمت : قیمت پایانی ← ( {percentage_diff:.2f} % ) ، آخرین قیمت ← ({percentage_akharin:.2f} %) | درصد نوسان پذیری ضمنی : {implied_vol * 100:.2f}", xref="paper", yref="paper", x=0.5, y=0.27, showarrow=False, font=dict(family="Vazirmatn FD ExtraBold, sans-serif", size=16, color="#c41768")),
                dict(text=f"اهرم (تاریخی/ضمنی) : {leverage_manual:.2f} / {leverage_implied:.2f} | دلتا (تاریخی/ضمنی)  : {delta_manual:.4f} / {delta_implied:.4f} | گاما (تاریخی/ضمنی) : {gamma_manual:.4f} / {gamma_implied:.4f} | تتا روزانه (تاریخی/ضمنی) : {theta_manual:.4f} / {theta_implied:.4f} | وگا (تاریخی/ضمنی) :  {vega_manual:.4f} / {vega_implied:.4f}", xref="paper", yref="paper", x=0.5, y=0.23, showarrow=False, font=dict(family="Vazirmatn FD ExtraBold, sans-serif", size=16, color="blue")),
                dict(text="Data_Bors : کانال تلگرام", align='center', xref="paper", yref="paper", x=1.02, y=0.5, textangle=-90, showarrow=False, font=dict(family="Vazirmatn FD ExtraBold, sans-serif", size=20, color="#3399ff"))
            ]
            fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=50, r=30, t=60, b=50), font=dict(family="Vazirmatn FD ExtraBold, sans-serif", size=14, color="RebeccaPurple"), annotations=clean_annotations, showlegend=False, yaxis_title="قیمت", yaxis2_title="حجم معاملات")
            file_path_final = os.path.join(swing_opportunities_folder, f'{nemad}.png')
            fig.write_image(file_path_final, width=1920, height=1080, scale=2)
            print(f" -> نمودار با موفقیت در '{file_path_final}' ذخیره شد.")
            send_to_telegram_api(file_path_final, telegram_caption)
            
    except Exception as e:
        if 'nemad' in locals():
            print(f"خطا در پردازش '{nemad}': {e}")
        else:
            print(f"خطا در پردازش یک آیتم: {e}")
        continue

print("\nاسکریپت به پایان رسید.")
