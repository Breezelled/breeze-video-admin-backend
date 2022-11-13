import re

import requests
from bs4 import BeautifulSoup
import unicodedata
import logging
import csv
import time
from lxml import etree
from selenium import webdriver
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
        # logging的配置，记录运行日志
        logging.basicConfig(filename="run.log", filemode="a+", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        # 表示当前处理的电影
        self.cur_movie_id = None
        self.cur_imdb_id = None

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
            # msg=1是URL失效，msg=2是电影没有海报，msg=3是电影没有预告片
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
        poster_url = ''
        # try:
        #     # 海报的高清大图URL
        #     poster_website_url = "https://www.imdb.com" + str(soup.find(
        #         class_='ipc-poster ipc-poster--baseAlt ipc-poster--dynamic-width sc-d383958-0 gvOdLN celwidget'
        #                ' ipc-sub-grid-item ipc-sub-grid-item--span-2').a['href'])
        #     # 访问
        #     response = self.get_url_response(poster_website_url)
        #     poster_soup = BeautifulSoup(response.content)
        #
        #     # print(poster_soup.find(class_="sc-7c0a9e7c-2 bkptFa").img['src'])
        #     for i in poster_soup.find_all(class_="sc-7c0a9e7c-2 bkptFa"):
        #         # 找到当前图片
        #         if "curr" in i.img["data-image-id"]:
        #             poster_url = i.img["src"]
        #     poster_re = self.get_url_response(poster_url)
        #     # 保存图片
        #     self.save_poster(self.cur_imdb_id, poster_re.content)
        # except AttributeError as e:
        #     # 如果没有海报链接，那么在黑名单中更新它
        #     # msg=3表示没有海报链接
        #     self.update_black_lst(self.cur_movie_id, '2')

        trailer_url = ''
        # try:
        #     # selenium打开网页
        #     driver.get(str(cur_url))
        #     driver.find_element(By.XPATH,
        #                         "/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[3]/div[1]"
        #                         "/div/div[2]/div[2]").click()
        #     trailer_website_url = driver.current_url
        #     # 访问
        #     response = self.get_url_response(trailer_website_url)
        #     trailer_soup = BeautifulSoup(response.content)
        #     # json解析
        #     trailer_url = json.loads(trailer_soup.find("script", {'id': '__NEXT_DATA__'}).get_text()).get("props")\
        #         .get("pageProps").get("videoPlaybackData").get("video").get("playbackURLs")[0].get("url")
        #
        #     print(trailer_url)
        #     trailer_re = self.get_url_response(trailer_url)
        #     # 保存预告片
        #     self.save_trailer(self.cur_imdb_id, trailer_re.content)
        # except AttributeError as e:
        #     # 如果没有预告片链接，那么在黑名单中更新它
        #     # msg=3表示没有预告片链接
        #     self.update_black_lst(self.cur_movie_id, '3')

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
        intro = soup.find(class_='sc-16ede01-6 cXGXRR').span.get_text().strip()
        intro = unicodedata.normalize('NFKC', intro)
        # 演员阵容 D W S，分别表示 导演，编剧，明星
        case_dict = {'D': [], 'W': [], 'S': []}
        for i in soup.find_all(class_='sc-bfec09a1-1 gfeYgX'):
            case_dict['S'].append(i.get_text())
        ul = soup.find(class_='ipc-metadata-list ipc-metadata-list--dividers-all sc-bfec09a1-8 jvByYy '
                                        'ipc-metadata-list--base')
        case_dict['D'].append(ul.li.div.ul.li.a.get_text())
        w_lis = ul.li.next_sibling.div.ul.children
        for i in w_lis:
            case_dict['W'].append(i.a.get_text())

        # id，电影名称，海报链接，时长，类型，发行时间，简介，导演，编剧，演员
        print(case_dict)
        detail = [self.cur_movie_id, name, poster_url, info[0], '|'.join(info[1:-1]),
                  info[-1], intro,
                  '|'.join(case_dict['D']), '|'.join(case_dict['W']), '|'.join(case_dict['S'])]
        self.save_info(detail)

    def save_poster(self, imdb_id, content):
        with open(f'{self.poster_save_path}/{imdb_id}.jpg', 'wb') as f:
            f.write(content)

    def save_trailer(self, imdb_id, content):
        with open(f'{self.trailer_save_path}/{imdb_id}.mp4', 'wb') as f:
            f.write(content)

    def save_info(self, detail):
        # 存储到CSV文件中
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
    # driver = webdriver.Chrome()
    s = Model()
    s.run()