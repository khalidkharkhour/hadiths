import logging
from flask import Flask, render_template, request
from utils import preprocess_text
from nlp_text_analysis import analyze_text

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    try:
        if request.method == 'POST':
            hadith_text = request.form.get('hadith_text', '').strip()
            logger.debug(f"Received hadith text: {hadith_text[:50]}...")
            if hadith_text:
                # Preprocess and analyze the text
                processed_text = preprocess_text(hadith_text)
                logger.info("Starting text analysis")
                analysis_results = analyze_text(processed_text)
                
                # Format results for display
                result = {
                    'summary': analysis_results.get('الملخص', 'لا يوجد ملخص'),
                    'context': {
                        'sentiment': analysis_results.get('السياق', {}).get('المشاعر', ''),
                        'confidence': analysis_results.get('السياق', {}).get('الثقة', ''),
                        'keywords': analysis_results.get('السياق', {}).get('الكلمات_المفتاحية', ''),
                        'inferred_context': analysis_results.get('السياق', {}).get('السياق_المستنتج', '')
                    },
                    'entities': {
                        'persons': analysis_results.get('الكيانات', {}).get('الأشخاص', []),
                        'locations': analysis_results.get('الكيانات', {}).get('الأماكن', []),
                        'dates': analysis_results.get('الكيانات', {}).get('التواريخ', []),
                        'organizations': analysis_results.get('الكيانات', {}).get('المنظمات', [])
                    },
                    'reliability': {
                        'contradictions': analysis_results.get('الموثوقية', {}).get('التناقضات', []),
                        'consistency': analysis_results.get('الموثوقية', {}).get('اتساق_التقييم', ''),
                        'reliability_note': analysis_results.get('الموثوقية', {}).get('ملاحظة_الموثوقية', '')
                    },
                    'hadith_comparison': {
                        'max_similarity': analysis_results.get('مقارنة_الأحاديث', {}).get('أقصى_التشابه', '0.00'),
                        'is_similar': analysis_results.get('مقارنة_الأحاديث', {}).get('مشابه', ''),
                        'authenticity_note': analysis_results.get('مقارنة_الأحاديث', {}).get('ملاحظة_الأصالة', ''),
                        'similar_hadiths': analysis_results.get('مقارنة_الأحاديث', {}).get('الأحاديث_المشابهة', [])
                    },
                    'analyse_scientifique':{
                    'scientific_analysis': analysis_results.get('analyse_scientifique', {})}

                }
                logger.info("Analysis completed successfully")
            else:
                result = {'error': '⚠️ الرجاء إدخال نص حديث.'}
                logger.warning("Empty hadith text submitted")
        return render_template('index.html', result=result)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        result = {'error': f'❌ خطأ أثناء المعالجة: {str(e)}'}
        return render_template('index.html', result=result)

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Failed to start Flask app: {str(e)}", exc_info=True)
        raise
