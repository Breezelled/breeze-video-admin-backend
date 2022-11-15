import re
import requests
from bs4 import BeautifulSoup
import unicodedata
import logging
import csv
import time
from selenium.webdriver.common.by import By
import json


class Model():
    def __init__(self):
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
                          " Version/15.4 Safari/605.1.15"
        }
        # 存放每一步电影的id和imdb的id
        self.movie_dct = {}
        # 存放已经处理完的movie id
        self.white_lst = []
        # 电影详情的初始url
        self.url = 'https://www.imdb.com/title/'
        self.movie_csv_path = '../ml-latest-small/links.csv'
        # 海报的保存路径
        self.poster_save_path = './poster'
        # 预告片的保存路径
        self.trailer_save_path = './trailer'
        # 电影信息的保存文件
        self.info_save_path = './info/info.csv'
        # 评论的保存文件
        self.reviews_save_path = './info/reviews.csv'
        # logging的配置，记录运行日志
        logging.basicConfig(filename="run.log", filemode="a+", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        # 表示当前处理的电影
        self.cur_movie_id = None
        self.cur_imdb_id = None
        # self.get_reviews_num = 5

    def get_white_lst(self):
        """获取处理完的白名单"""
        with open(r'white_list') as f:
            for line in f:
                line = line.strip()
                self.white_lst.append(line)

    def get_movie_id(self):
        """获取电影的id和imdb的id"""
        with open(self.movie_csv_path) as f:
            f.readline()
            for line in f:
                line = line.strip()
                line = line.split(',')
                # 电影id 对应 imdbid
                self.movie_dct[line[0]] = line[1]

    def update_white_lst(self, movie_id):
        """更新白名单"""
        with open(r'white_list', 'a+') as f:
            f.write(movie_id + '\n')

    def update_black_lst(self, movie_id, msg=''):
        with open(r'black_list', 'a+') as f:
            # 写入movie id 和imdb id，并且加上错误原因
            # msg=1是URL失效，msg=2是电影没有海报，msg=3是电影没有预告片，msg=4是电影没有评论
            f.write(movie_id + ' ' + self.movie_dct[movie_id] + ' ' + msg + '\n')

    def get_url_response(self, url):
        """访问网页请求，返回response"""
        logging.info(f'get {url}')
        i = 0
        # 超时重传，最多5次
        while i < 5:
            try:
                response = requests.get(url, timeout=6)
                if response.status_code == 200:
                    logging.info(f'get {url} sucess')
                    # 正常获取，直接返回
                    return response
                # 如果状态码不对，获取失败，返回None，不再尝试
                logging.error(f'get {url} status_code error: {response.status_code} movie_id is {self.cur_movie_id}')
                return None
            except requests.RequestException:
                # 如果超时
                logging.error(f'get {url} error, try to restart {i + 1}')
                i += 1
        # 重试5次都失败，返回None
        return None

    def process_html(self, html, cur_url):
        """解析html，获取海报，电影信息"""
        soup = BeautifulSoup(html, 'lxml')
        # 名字和发布日期 如：Toy Story (1995)
        name = soup.find(attrs={'class': re.compile('^sc-b73cd867-0')}).string
        print(type(name))
        # 去掉html的一些/x20等空白符
        name = unicodedata.normalize('NFKC', name)
        print(name)

        # 电影的基本信息   1h 21min | Animation, Adventure, Comedy | 21 March 1996 (Germany)
        info = []
        try:
            # 时长时间
            info.append(soup.find("ul", class_='ipc-metadata-list ipc-metadata-list--dividers-none ipc-metadata-list--'
                                               'compact ipc-metadata-list--base').li.div.get_text())
        except AttributeError as e:
            # 没有则添加空字符串
            info.append('')

        # 基本信息和详细发布时间 Animation, Adventure, Comedy | 21 March 1996 (Germany)
        for tag in soup.find(class_='ipc-chip-list__scroller').find_all('a'):
            info.append(tag.get_text().strip())
        release_date_li = soup.find("li", attrs={"class": "ipc-metadata-list__item ipc-metadata-list-item--link",
                                                 "data-testid": "title-details-releasedate"})
        info.append(release_date_li.div.get_text())
        print(info)
        # 简介
        intro = soup.find(attrs={'class': re.compile('^sc-16ede01-0')}).get_text().strip()
        intro = unicodedata.normalize('NFKC', intro)
        # 演员阵容 D W S，分别表示 导演，编剧，明星
        try:
            case_dict = {'D': [], 'W': [], 'S': []}
            for i in soup.find_all(attrs={"class": re.compile('^sc-bfec09a1-1')}):
                case_dict['S'].append(i.get_text())
            ul = soup.find(attrs={"class": re.compile('^ipc-metadata-list ipc-metadata-list--dividers-all sc-bfec09a1-8')})
            case_dict['D'].append(ul.li.div.ul.li.a.get_text())
            w_lis = ul.li.next_sibling.div.ul.children
            for i in w_lis:
                case_dict['W'].append(i.a.get_text())
        except AttributeError as e:
            pass


        budget_gross = []

        # 预算和票房
        try:
            budget_gross.append(soup.find(attrs={"class": "ipc-metadata-list__item sc-6d4f3f8c-2 fJEELB",
                                                 "data-testid": "title-boxoffice-budget"}).div.get_text())
        except AttributeError as e:
            budget_gross.append('')

        try:
            budget_gross.append(soup.find(attrs={"class": "ipc-metadata-list__item sc-6d4f3f8c-2 fJEELB",
                                                 "data-testid": "title-boxoffice-cumulativeworldwidegross"})
                                .div.get_text())
        except AttributeError as e:
            budget_gross.append('')

        print(budget_gross)

        lang, comp, country = [], [], []
        # 影片语言
        try:
            lang_ul = soup.find(attrs={"class": "ipc-metadata-list__item", "data-testid": "title-details-languages"}) \
                .div.ul
            for i in lang_ul.children:
                lang.append(i.get_text())
        except AttributeError as e:
            pass
        # 制作公司
        try:
            comp_ul = soup.find(attrs={"class": "ipc-metadata-list__item ipc-metadata-list-item--link",
                                       "data-testid": "title-details-companies"}).div.ul
            for i in comp_ul.children:
                comp.append(i.get_text())
        except AttributeError as e:
            pass
        # 制作国家
        try:
            country_ul = soup.find(attrs={"class": "ipc-metadata-list__item",
                                          "data-testid": "title-details-origin"}).div.ul
            for i in country_ul.children:
                country.append(i.get_text())
        except AttributeError as e:
            pass

        rating, review_num, tagline = '', '', ''
        # 评分，评分数，标语
        try:
            rating = soup.find(attrs={"class": re.compile('^sc-7ab21ed2-2')}).get_text()
        except AttributeError as e:
            pass

        try:
            review_num = soup.find(attrs={"class": re.compile('^sc-7ab21ed2-3')}).get_text()
        except AttributeError as e:
            pass

        try:
            response = self.get_url_response(cur_url + '/taglines')
            tagline_soup = BeautifulSoup(response.content, "lxml")
            tagline = tagline_soup.find("div", class_="soda odd").get_text().strip()
        except AttributeError as e:
            pass
        print(tagline)
        # id，电影名称，时长，类型，发行时间，简介，导演，编剧，演员，预算，票房，语言，制作公司，制作国家，评分，评分数，标语
        print(case_dict)
        detail = [self.cur_movie_id, name, info[0], '|'.join(info[1:-1]),
                  info[-1], intro, '|'.join(case_dict['D']), '|'.join(case_dict['W']), '|'.join(case_dict['S']),
                  budget_gross[0], budget_gross[1], '|'.join(lang), '|'.join(comp), '|'.join(country), rating,
                  review_num, tagline]
        print(detail)
        self.save_info(detail)

        # 评论
        # T是标题 C是内容 A是作者 D是时间
        reviews_dict = {'T': [], 'C': [], 'A': [], 'D': []}
        try:
            response = self.get_url_response(cur_url + '/reviews')
            review_soup = BeautifulSoup(response.content, "lxml")
            title = review_soup.findAll("a", class_="title")
            content = review_soup.findAll("div", class_="text show-more__control")
            author = review_soup.findAll("span", class_="display-name-link")
            date = review_soup.findAll("span", class_="review-date")
            i = 0
            get_reviews_num = 5
            if get_reviews_num > len(title):
                get_reviews_num = len(title)
            while i != get_reviews_num:
                reviews_dict['T'].append(title[i].get_text().strip())
                reviews_dict['C'].append(unicodedata.normalize('NFKC', content[i].get_text()))
                reviews_dict['A'].append(author[i].a.get_text())
                reviews_dict['D'].append(date[i].get_text())
                i += 1

        except AttributeError as e:
            # 如果没有评论链接，那么在黑名单中更新它
            # msg=4表示没有评论链接
            self.update_black_lst(self.cur_movie_id, '4')
        print(reviews_dict)
        # id，作者，时间，标题，内容
        with open(f'{self.reviews_save_path}', 'a+', encoding='utf-8', newline='') as f:
            for i in range(len(reviews_dict['T'])):
                review = [self.cur_movie_id, reviews_dict['A'][i], reviews_dict['D'][i],
                          reviews_dict['T'][i], reviews_dict['C'][i]]
                writer = csv.writer(f)
                writer.writerow(review)

    def save_poster(self, imdb_id, content):
        with open(f'{self.poster_save_path}/{imdb_id}.jpg', 'wb') as f:
            f.write(content)

    def save_trailer(self, imdb_id, content):
        with open(f'{self.trailer_save_path}/{imdb_id}.mp4', 'wb') as f:
            f.write(content)

    def save_info(self, detail):
        with open(f'{self.info_save_path}', 'a+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(detail)

    def run(self):
        # 开始爬取信息
        # 先读入文件
        self.get_white_lst()
        self.get_movie_id()
        for movie_id, imdb_id in self.movie_dct.items():
            if movie_id in self.white_lst:
                continue
            self.cur_movie_id = movie_id
            self.cur_imdb_id = imdb_id
            time.sleep(1)
            cur_url = self.url + 'tt' + self.cur_imdb_id
            response = self.get_url_response(cur_url)
            # 找不到电影详情页的url，或者超时，则仅仅保留id，之后再用另一个脚本处理
            if response is None:
                self.save_info([self.cur_movie_id, '' * 9])
                # 仍然更新白名单，避免重复爬取这些失败的电影
                self.update_white_lst(self.cur_movie_id)
                # 更新黑名单，爬完之后用另一个脚本再处理
                self.update_black_lst(self.cur_movie_id, '1')
                continue
            # 处理电影详情信息
            self.process_html(response.content, cur_url)
            # 处理完成，增加movie id到白名单中
            self.update_white_lst(self.cur_movie_id)
            logging.info(f'process movie {self.cur_movie_id} success')


if __name__ == '__main__':
    # selenium
    s = Model()
    s.run()
