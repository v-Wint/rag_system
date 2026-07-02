from pydantic import BaseModel

class SchemaNode(BaseModel):
    title: str = ''
    children: list['SchemaNode'] = []

    def _to_str(self, depth=0):
        lines = []
        if self.title:
            lines.append('    ' * depth + '- ' + self.title)
        for child in self.children:
            lines.append(child._to_str(depth + 1))
        return '\n'.join(lines)

    def __bool__(self) -> bool:
        return bool(self.title or self.children)
    def __str__(self):
        return self._to_str()

SchemaNode.model_rebuild()
