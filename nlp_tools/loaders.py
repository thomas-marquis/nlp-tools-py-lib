import re
import typing as t


class InputData:
    examples: t.Dict[str, t.List[str]]
    responses: t.Dict[str, t.List[str]]

    def __init__(self, examples, responses):
        self.examples = examples
        self.responses = responses


class DataLoader:
    def load(self) -> dict:
        raise Exception('load method not implemented')


class MdLoader(DataLoader):
    path: str

    def __init__(self, path: str):
        DataLoader.__init__(self)
        self.path = path

    def get_if_match(self, regex, text):
        matcher = re.match(regex, text)
        if matcher is None:
            return None
        return matcher.group(1)

    def get_list_item(self, line):
        return self.get_if_match(r'- (.*)$', line)

    def get_intent_name(self, line):
        return self.get_if_match(r'^## (.*)$', line)

    def get_sub_part(self, line: str) -> str:
        return self.get_if_match(r'^### (.*)$', line)

    def load(self) -> InputData:
        with open(self.path, 'r', encoding='utf-8') as file:
            content = [re.sub('\\n', '', l) for l in file]
        mega_part = ''
        current_intent = ''
        current_sub_part = ''
        intents = dict()
        response = dict()

        for line in content:
            if line == '# intents':
                mega_part = 'intents'
            if mega_part == 'intents':
                intent_name: str = self.get_intent_name(line)
                list_item: str = self.get_list_item(line)
                sub_part: str = self.get_sub_part(line)

                if intent_name is not None and intent_name != '':
                    current_intent = intent_name
                    if intent_name != 'not_found':
                        intents[intent_name] = []
                    response[intent_name] = []

                if sub_part is not None:
                    current_sub_part = sub_part

                if current_sub_part == 'examples' and list_item is not None:
                    if current_intent != 'not_found':
                        intents[current_intent].append(list_item)

                if current_sub_part == 'response' and list_item is not None:
                    response[current_intent].append(list_item)

        return InputData(intents, response)
