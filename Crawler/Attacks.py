import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href,
                           callback=self.parse_jutsu)
            yield extracted_data

        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)
    
    def parse_jutsu(self, response):
        attack_name = response.css("span.mw-page-title-main::text").extract()[0]
        attack_name = attack_name.strip()

        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')

        attack_type=""
        if soup.find('aside'):
            aside = soup.find('aside')

            for cell in aside.find_all('div',{'class':'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Classification":
                        attack_type = cell.find('div').text.strip()

        soup.find('aside').decompose()

        Attack_description = soup.text.strip()
        Attack_description = Attack_description.split('Trivia')[0].strip()

        return dict (
            attack_name = attack_name,
            attack_type = attack_type,
            Attack_description = Attack_description
        )