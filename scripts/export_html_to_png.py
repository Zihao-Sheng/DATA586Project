import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def export_html_dir_to_png(html_dir: Path):
    html_files = sorted(html_dir.glob('*.html'))
    if not html_files:
        print('No html files found')
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1900, "height": 1200})

        ok = 0
        fail = 0
        for html_path in html_files:
            png_path = html_path.with_suffix('.png')
            url = html_path.resolve().as_uri()
            try:
                page.goto(url, wait_until='networkidle', timeout=90000)
                page.wait_for_timeout(1200)

                plot = page.locator('.js-plotly-plot').first
                if plot.count() > 0:
                    plot.screenshot(path=str(png_path))
                else:
                    page.screenshot(path=str(png_path), full_page=True)
                ok += 1
                print(f'OK  {html_path.name} -> {png_path.name}')
            except Exception as e:
                fail += 1
                print(f'ERR {html_path.name}: {e}')

        browser.close()

    print(f'Done. ok={ok}, fail={fail}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='exports/experiment_analysis_exports/graphs')
    args = parser.parse_args()
    export_html_dir_to_png(Path(args.dir))
